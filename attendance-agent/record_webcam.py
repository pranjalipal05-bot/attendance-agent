"""Record a test video from your webcam (video only):  python record_webcam.py
Press 'q' in the preview window to stop. Saved to data/webcam_test.mp4"""
import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open the webcam.")
w, h = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter("data/webcam_test.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
print("Recording... press 'q' to stop.")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    out.write(frame)
    cv2.imshow("Recording (q to stop)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release(); out.release(); cv2.destroyAllWindows()
print("Saved data/webcam_test.mp4")
