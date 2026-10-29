"""
Module 1 - Automated attendance tracking (computer vision).

* Detects faces with OpenCV Haar cascades (frontal = facing the front,
  profile = looking away). Runs fully offline - no cloud, no face recognition,
  no stored images. People are only anonymous IDs ("Person 1", "Person 2"...).
* A simple centroid tracker keeps the same ID for a person across frames.
* Attendance is marked only after SUSTAINED presence (default 5 s), so a
  one-frame false detection or someone walking past is not counted.
"""
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from utils import fmt_time

FRONTAL = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
PROFILE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")


def detect_faces(gray):
    """Return a list of (x, y, w, h, facing). facing=True for frontal faces."""
    gray = cv2.equalizeHist(gray)
    img_w = gray.shape[1]

    dets = [(int(x), int(y), int(w), int(h), True)
            for (x, y, w, h) in FRONTAL.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))]

    profiles = [tuple(b) for b in PROFILE.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))]
    flipped = cv2.flip(gray, 1)  # profile cascade only sees one side, so flip too
    for (x, y, w, h) in PROFILE.detectMultiScale(flipped, 1.1, 5, minSize=(30, 30)):
        profiles.append((img_w - x - w, y, w, h))

    for (x, y, w, h) in profiles:
        cx, cy = x + w / 2, y + h / 2
        duplicate = any(abs(cx - (fx + fw / 2)) < fw * 0.7 and abs(cy - (fy + fh / 2)) < fh * 0.7
                        for fx, fy, fw, fh, _ in dets)
        if not duplicate:
            dets.append((int(x), int(y), int(w), int(h), False))
    return dets


class Tracker:
    """Greedy nearest-centroid tracker that gives each person a stable ID."""

    def __init__(self, max_missing_s=4.0):
        self.tracks = {}
        self.next_id = 1
        self.max_missing = max_missing_s

    def update(self, dets, t):
        """Return {detection_index: (track_id, motion)}. Motion is displacement
        since the last sample, measured in face-widths (so it is scale-free)."""
        for tid in [k for k, v in self.tracks.items() if t - v["last_t"] > self.max_missing]:
            del self.tracks[tid]

        pairs = []
        for i, (x, y, w, h, _) in enumerate(dets):
            cx, cy = x + w / 2, y + h / 2
            for tid, tr in self.tracks.items():
                d = float(np.hypot(cx - tr["c"][0], cy - tr["c"][1]))
                if d < max(w, tr["w"]) * 1.5:
                    pairs.append((d, i, tid))
        pairs.sort()

        used_det, used_trk, result = set(), set(), {}
        for d, i, tid in pairs:
            if i in used_det or tid in used_trk:
                continue
            used_det.add(i)
            used_trk.add(tid)
            x, y, w, h, _ = dets[i]
            self.tracks[tid].update(c=(x + w / 2, y + h / 2), w=w, last_t=t)
            result[i] = (tid, d / max(w, 1))

        for i, (x, y, w, h, _) in enumerate(dets):
            if i not in used_det:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"c": (x + w / 2, y + h / 2), "w": w, "last_t": t}
                result[i] = (tid, 0.0)
        return result


def detect_people(video_path, sample_fps=4, progress_cb=None):
    """Scan the video and return (records_df, duration_s, snapshots, sample_dt).

    records_df has one row per person per sampled frame:
        time_s, person, facing (bool), motion (face-widths moved)
    snapshots are 3 annotated RGB frames for the dashboard.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not open the video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / sample_fps)))
    sample_dt = step / fps
    snap_at = {int(max(1, total // step) * f) for f in (0.2, 0.5, 0.8)}

    tracker, rows, snaps = Tracker(), [], []
    idx = sample_no = 0
    while cap.grab():
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                t = idx / fps
                h, w = frame.shape[:2]
                if w > 640:
                    frame = cv2.resize(frame, (640, int(h * 640 / w)))
                dets = detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                matched = tracker.update(dets, t)
                for i, (tid, motion) in matched.items():
                    rows.append({"time_s": t, "person": f"Person {tid}",
                                 "facing": dets[i][4], "motion": motion})
                if sample_no in snap_at:
                    snaps.append(_annotate(frame, dets, matched))
                sample_no += 1
                if progress_cb and total and sample_no % 10 == 0:
                    progress_cb(min(1.0, idx / total))
        idx += 1
    cap.release()

    duration = idx / fps
    records = pd.DataFrame(rows, columns=["time_s", "person", "facing", "motion"])
    return records, duration, snaps, sample_dt


def _annotate(frame, dets, matched):
    img = frame.copy()
    for i, (x, y, w, h, facing) in enumerate(dets):
        color = (0, 200, 0) if facing else (0, 165, 255)  # green = facing, orange = away
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        if i in matched:
            cv2.putText(img, f"P{matched[i][0]}", (x, max(15, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_attendance(records, min_presence_s, sample_dt, session_start=None):
    """Timestamped attendance log based on sustained presence."""
    session_start = session_start or datetime.now()
    rows = []
    for person, g in records.groupby("person"):
        g = g.sort_values("time_s")
        visible = len(g) * sample_dt
        present = visible >= min_presence_s
        marked_s = None
        if present:
            k = int(np.ceil(min_presence_s / sample_dt)) - 1
            marked_s = float(g["time_s"].iloc[min(k, len(g) - 1)])
        rows.append({
            "Person": person,
            "First seen": fmt_time(g["time_s"].min()),
            "Marked present at": fmt_time(marked_s),
            "Clock time": (session_start + timedelta(seconds=marked_s)).strftime("%H:%M:%S")
                          if present else "-",
            "Time visible (s)": round(visible, 1),
            "Status": "Present" if present else "Not counted (brief detection)",
            "_first": float(g["time_s"].min()),
        })
    log = pd.DataFrame(rows).sort_values("_first").drop(columns="_first")
    return log.reset_index(drop=True)
