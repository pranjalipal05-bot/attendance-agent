# Conclusion

The Multimodal Attendance & Engagement Agent shows that attendance and engagement can be
tracked automatically, offline and without identifying individuals. Computer vision marks
attendance from sustained presence, audio processing adds interaction context, and a hybrid of
transparent rules and a RandomForest model produces an engagement timeline. A rule-based agent
turns that timeline into practical suggestions for instructors, and a Streamlit dashboard
presents attendance, engagement, suggestions and per-student analytics.

## Limitations
- Haar cascades work best with frontal, well-lit faces; occlusion and small faces reduce accuracy.
- Head pose is approximated as facing front vs. looking away.
- The bundled ML model is bootstrapped on synthetic data; accuracy on real classrooms needs labelled clips.
- A person who leaves and returns after several seconds may receive a new anonymous ID.

## Future work
- MediaPipe Face Mesh for true head-pose angles
- Training on labelled real classroom data (`data/labeled_features.csv`)
- Multi-camera support and live webcam analysis
- LMS integration and exportable session reports
