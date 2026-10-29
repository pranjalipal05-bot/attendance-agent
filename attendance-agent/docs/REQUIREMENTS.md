# Epic 1 - Requirements Specification and Environment Setup

## Goal
An offline, privacy-preserving classroom system that automatically tracks attendance,
analyses engagement from video and audio, and gives instructors AI-driven teaching suggestions.

## Functional requirements
| ID | Requirement |
|----|-------------|
| FR1 | Process a recorded (or webcam) classroom video |
| FR2 | Detect and track participants; mark attendance only after sustained presence |
| FR3 | Generate a timestamped attendance log (CSV) |
| FR4 | Measure head direction, movement, inactivity and audio activity |
| FR5 | Fuse signals into an engagement score per time window |
| FR6 | Generate context-aware teaching suggestions |
| FR7 | Provide per-student session analytics |
| FR8 | Offer a dashboard to view and download results |

## Non-functional requirements
- Runs fully offline; no cloud services
- Privacy: no face recognition, no identities, no stored images or speech
- Runs on a normal laptop (Core i5 / Ryzen 5, 8 GB RAM)
- Suggestions are assistive only; the instructor decides

## Environment
- Python 3.8+, VS Code, Git, Chrome/Edge/Firefox
- Windows 10/11, macOS Monterey+, or Ubuntu 20.04+
- Libraries: OpenCV, NumPy, pandas, SciPy, scikit-learn, imageio-ffmpeg, Streamlit

## Setup
```
python -m venv venv
venv\Scripts\activate        (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## Architecture
```
video -> attendance.py (faces + tracker) --> attendance log
      -> audio.py (loudness per second) --+
                                          v
              engagement.py (rules + ml_model.py RandomForest) -> engagement timeline
                                          v
              suggestions.py (agent rules) -> teaching suggestions
                                          v
                        app.py (Streamlit dashboard)
```
