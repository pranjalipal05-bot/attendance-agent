"""
Module 2b - Multimodal engagement analysis.

Fuses three signals into one 0-100 engagement score per time window:
    attention  - share of samples where the person faces the front (head pose proxy)
    movement   - share of samples with visible movement (inactivity lowers it)
    audio      - share of seconds with audio activity (class interaction)

These are transparent heuristics, not clinical measurements; the weights and
thresholds below can be tuned.
"""
import numpy as np
import pandas as pd

from ml_model import predict_engagement
from utils import fmt_time

MOTION_THRESH = 0.08          # face-widths per sample that counts as "moving"
W_ATT, W_MOVE, W_AUDIO = 0.6, 0.2, 0.2
W_ATT_NO_AUDIO, W_MOVE_NO_AUDIO = 0.75, 0.25


def _score(attention, moving_ratio, audio_ratio):
    movement = min(1.0, moving_ratio * 2)   # moving half the time = full marks
    if audio_ratio is None:
        return 100 * (W_ATT_NO_AUDIO * attention + W_MOVE_NO_AUDIO * movement)
    audio = min(1.0, audio_ratio * 1.5)
    return 100 * (W_ATT * attention + W_MOVE * movement + W_AUDIO * audio)


def compute_engagement(records, duration, audio_active, window_s=30, model=None):
    """Return (windows_df, person_windows_df).

    windows_df: one row per time window for the whole class.
    person_windows_df: one row per person per window (used for student analytics).
    """
    win_rows, pw_rows = [], []
    n_windows = max(1, int(np.ceil(duration / window_s)))
    for k in range(n_windows):
        t0, t1 = k * window_s, min(duration, (k + 1) * window_s)
        sub = records[(records["time_s"] >= t0) & (records["time_s"] < t1)]

        a_ratio = None
        if audio_active is not None:
            seg = audio_active[int(t0):int(np.ceil(t1))]
            a_ratio = float(np.mean(seg)) if len(seg) else 0.0

        if sub.empty:
            continue
        scores, atts, movs = [], [], []
        for person, g in sub.groupby("person"):
            att = float(g["facing"].mean())
            mov = float((g["motion"] > MOTION_THRESH).mean())
            s = _score(att, mov, a_ratio)
            if model is not None:                      # blend rules with the ML prediction
                s = 0.5 * s + 0.5 * predict_engagement(model, att, min(1.0, mov * 2), a_ratio)
            scores.append(s); atts.append(att); movs.append(mov)
            pw_rows.append({"person": person, "window": k, "attention": att,
                            "movement": mov, "engagement": s})
        win_rows.append({
            "window": k, "start_s": t0, "end_s": t1,
            "label": f"{fmt_time(t0)}-{fmt_time(t1)}",
            "people": len(scores),
            "attention": round(100 * np.mean(atts), 1),
            "movement": round(100 * np.mean(movs), 1),
            "audio": None if a_ratio is None else round(100 * a_ratio, 1),
            "engagement": round(float(np.mean(scores)), 1),
        })
    return pd.DataFrame(win_rows), pd.DataFrame(pw_rows)


def student_analytics(records, person_windows, sample_dt, attendance):
    """Per-student session summary."""
    rows = []
    for person, g in records.sort_values("time_s").groupby("person"):
        # longest stretch without visible movement
        longest = cur = 0
        prev_t = None
        for t, m in zip(g["time_s"], g["motion"]):
            contiguous = prev_t is not None and (t - prev_t) <= 1.5 * sample_dt
            if m < MOTION_THRESH and (contiguous or prev_t is None):
                cur += 1
            elif m < MOTION_THRESH:
                cur = 1
            else:
                cur = 0
            longest = max(longest, cur)
            prev_t = t
        pw = person_windows[person_windows["person"] == person]
        rows.append({
            "Person": person,
            "Time present (s)": round(len(g) * sample_dt, 1),
            "Avg engagement": round(float(pw["engagement"].mean()), 1) if len(pw) else 0.0,
            "Attention (% facing front)": round(100 * float(g["facing"].mean()), 1),
            "Longest inactivity (s)": round(longest * sample_dt, 1),
        })
    return pd.DataFrame(rows)
