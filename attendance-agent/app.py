"""
Multimodal Attendance & Engagement Agent - Streamlit dashboard.
Run:  streamlit run app.py
"""
import tempfile

import streamlit as st

from pipeline import run_analysis
from utils import fmt_time

st.set_page_config(page_title="Attendance & Engagement Agent", layout="wide")
st.title("Multimodal Attendance & Engagement Agent")
st.caption("Offline, privacy-preserving classroom intelligence: attendance, engagement "
           "and teaching suggestions. No cloud, no face recognition, no stored images.")

with st.sidebar:
    st.header("Settings")
    video = st.file_uploader("Classroom video", type=["mp4", "avi", "mov", "mkv"])
    min_presence = st.slider("Seconds present to mark attendance", 2, 30, 5)
    window = st.select_slider("Engagement window (seconds)", options=[10, 15, 30, 60], value=30)
    fps = st.slider("Analysis speed (samples per second)", 1, 8, 4,
                    help="Lower = faster but less detailed")
    run = st.button("Analyze session", type="primary", disabled=video is None)

if video is None:
    st.info("Upload a recorded classroom video in the sidebar to begin. "
            "Clear, front-facing faces and a 1-5 minute clip work best.")
    st.stop()

if run:
    suffix = "." + video.name.rsplit(".", 1)[-1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(video.read())
        path = f.name
    bar = st.progress(0.0, text="Detecting and tracking participants...")
    st.session_state["res"] = run_analysis(path, fps, min_presence, window,
                                           progress_cb=lambda p: bar.progress(p, text="Analyzing video..."))
    st.session_state["window_s"] = window
    bar.empty()

res = st.session_state.get("res")
if not res:
    st.stop()
if res["empty"]:
    st.warning("No participants were detected for long enough. Try a clearer video, "
               "or lower 'Seconds present' in the sidebar.")
    st.stop()

att, win, stu = res["attendance"], res["windows"], res["students"]
present = int((att["Status"] == "Present").sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Participants present", present)
c2.metric("Session length", fmt_time(res["duration"]))
c3.metric("Average engagement", f"{win['engagement'].mean():.0f} / 100")
c4.metric("Audio detected", "Yes" if res["has_audio"] else "No")
st.caption("Scoring: " + ("rules + machine-learning model (RandomForest)" if res.get("ml_used") else "rule-based"))

tab1, tab2, tab3, tab4 = st.tabs(["Attendance", "Engagement", "Teaching suggestions", "Student analytics"])

with tab1:
    st.subheader("Timestamped attendance log")
    st.dataframe(att, use_container_width=True)
    st.download_button("Download attendance CSV", att.to_csv(index=False), "attendance_log.csv")
    if res["snapshots"]:
        st.subheader("Detection samples (green = facing front, orange = looking away)")
        cols = st.columns(len(res["snapshots"]))
        for col, img in zip(cols, res["snapshots"]):
            col.image(img)

with tab2:
    st.subheader("Engagement over the session")
    chart = win.set_index("label")[["engagement", "attention", "movement"]]
    if win["audio"].notna().any():
        chart["audio"] = win.set_index("label")["audio"]
    st.line_chart(chart)
    st.caption("Engagement = attention (head pose) + movement/inactivity + audio activity, scaled 0-100.")
    st.dataframe(win.drop(columns=["window", "start_s", "end_s"]), use_container_width=True)
    st.download_button("Download engagement CSV", win.to_csv(index=False), "engagement_timeline.csv")

with tab3:
    st.subheader("AI-driven teaching suggestions")
    for kind, text in res["suggestions"]:
        {"good": st.success, "warn": st.warning}.get(kind, st.info)(text)

with tab4:
    st.subheader("Per-student session analytics (anonymous IDs)")
    st.dataframe(stu, use_container_width=True)
    st.bar_chart(stu.set_index("Person")["Avg engagement"])
    st.download_button("Download student CSV", stu.to_csv(index=False), "student_analytics.csv")
