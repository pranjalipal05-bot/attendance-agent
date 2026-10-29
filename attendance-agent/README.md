# Multimodal Attendance & Engagement Agent

Offline, privacy-preserving classroom intelligence in Python: automatic attendance, multimodal
engagement analysis, AI-driven teaching suggestions and student session analytics.

## Project structure (maps to the 6 epics)
| Epic | Files |
|---|---|
| 1. Requirements & setup | `docs/REQUIREMENTS.md`, `requirements.txt` |
| 2. Vision module | `attendance.py` |
| 3. Audio module | `audio.py` |
| 4. AI implementation | `ml_model.py`, `train_model.py`, `engagement.py` |
| 5. Analytics & recommendations | `engagement.py` (student analytics), `suggestions.py` |
| 6. Interface, testing, deployment | `app.py`, `pipeline.py`, `tests/`, `.github/workflows/tests.yml` |
| Conclusion | `docs/CONCLUSION.md` |

## Setup
```
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Run
```
streamlit run app.py                   # dashboard: upload a video in the sidebar
python pipeline.py data/class.mp4      # command line; saves CSVs in outputs/
python record_webcam.py                # optional: record a test video from your webcam
python train_model.py                  # optional: retrain the ML model
python -m unittest discover tests -v   # run the tests
```
The ML model trains automatically on first run if it is missing.

## Tips for a good test video
- 1-3 minutes, faces clearly visible and well lit (2-3 people is ideal).
- Have people look away or lean back for a while so the graph shows drops and peaks.
- Speak or ask a question for part of the video so audio activity appears.

## How it works
1. **Detect + track:** OpenCV Haar cascades find frontal faces (facing front) and profile faces
   (looking away); a centroid tracker assigns anonymous IDs.
2. **Attendance:** a person is marked present after `Seconds present` (default 5 s) of visibility.
3. **Engagement score (per window):** rule score `0.6*attention + 0.2*movement + 0.2*audio`,
   blended 50/50 with a RandomForest probability from `ml_model.py`.
4. **Suggestions:** rules detect drops, peaks, fading attention, low interaction and participants leaving.

## Train on your own labels (optional)
Create `data/labeled_features.csv` with columns `attention,movement,audio,has_audio,engaged`
(values 0-1; `engaged` is 0 or 1), then run `python train_model.py`.

## Privacy
Runs fully offline. No face recognition, no identities, no images or speech stored.

## License
MIT - edit `LICENSE` and put your name in it.
