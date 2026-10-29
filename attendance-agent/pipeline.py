"""
Glue code: video -> attendance -> engagement -> suggestions.
Run from the command line:   python pipeline.py data/class.mp4
"""
import os
import sys
from datetime import datetime

from attendance import detect_people, build_attendance
from audio import audio_activity
from engagement import compute_engagement, student_analytics
from ml_model import load_model
from suggestions import generate_suggestions


def run_analysis(video_path, sample_fps=4, min_presence_s=5, window_s=30, progress_cb=None):
    records, duration, snaps, sample_dt = detect_people(video_path, sample_fps, progress_cb)
    result = {"duration": duration, "snapshots": snaps, "records": records}
    if records.empty:
        result["empty"] = True
        return result

    attendance = build_attendance(records, min_presence_s, sample_dt, datetime.now())
    counted = attendance.loc[attendance["Status"] == "Present", "Person"]
    records = records[records["person"].isin(counted)]
    if records.empty:
        result.update(empty=True, attendance=attendance)
        return result

    audio = audio_activity(video_path)
    model = load_model()                      # None -> rule-based scoring only
    windows, person_windows = compute_engagement(records, duration, audio, window_s, model)
    students = student_analytics(records, person_windows, sample_dt, attendance)
    tips = generate_suggestions(windows, students, window_s)

    result.update(empty=False, attendance=attendance, windows=windows,
                  students=students, suggestions=tips, has_audio=audio is not None,
                  ml_used=model is not None)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python pipeline.py <video_file>")
    res = run_analysis(sys.argv[1], progress_cb=lambda p: print(f"\rProcessing {p:5.0%}", end=""))
    print()
    if res["empty"]:
        sys.exit("No people were detected for long enough. Use a video with clearly visible faces.")
    os.makedirs("outputs", exist_ok=True)
    res["attendance"].to_csv("outputs/attendance_log.csv", index=False)
    res["windows"].to_csv("outputs/engagement_timeline.csv", index=False)
    res["students"].to_csv("outputs/student_analytics.csv", index=False)
    print(res["attendance"].to_string(index=False))
    print("\nSuggestions:")
    for kind, text in res["suggestions"]:
        print(f" [{kind}] {text}")
    print("\nSaved CSVs in ./outputs")
