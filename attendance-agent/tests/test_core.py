"""Unit tests (Epic 6). Run from the project root:  python -m unittest discover tests -v
(also works with pytest)"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attendance import Tracker, build_attendance, detect_faces
from engagement import compute_engagement, student_analytics
from ml_model import predict_engagement, train_model
from suggestions import generate_suggestions


def make_records(person, start, end, dt=0.25, facing=True, motion=0.2):
    times = np.arange(start, end, dt)
    return pd.DataFrame({"time_s": times, "person": person, "facing": facing, "motion": motion})


class TestAttendance(unittest.TestCase):
    def test_sustained_presence_required(self):
        recs = pd.concat([make_records("Person 1", 0, 20), make_records("Person 2", 3, 4)])
        log = build_attendance(recs, min_presence_s=5, sample_dt=0.25).set_index("Person")
        self.assertEqual(log.loc["Person 1", "Status"], "Present")
        self.assertTrue(log.loc["Person 2", "Status"].startswith("Not counted"))

    def test_marked_after_threshold(self):
        log = build_attendance(make_records("Person 1", 0, 20), 5, 0.25)
        self.assertEqual(log.loc[0, "Marked present at"], "00:05")

    def test_tracker_keeps_same_id(self):
        tr = Tracker()
        first = tr.update([(100, 100, 50, 50, True)], 0.0)
        second = tr.update([(104, 102, 50, 50, True)], 0.25)
        self.assertEqual(first[0][0], second[0][0])

    def test_tracker_new_id_for_new_person(self):
        tr = Tracker()
        tr.update([(100, 100, 50, 50, True)], 0.0)
        out = tr.update([(100, 100, 50, 50, True), (400, 100, 50, 50, True)], 0.25)
        self.assertEqual(len({v[0] for v in out.values()}), 2)

    def test_blank_frame_has_no_faces(self):
        self.assertEqual(detect_faces(np.zeros((240, 320), np.uint8)), [])


class TestEngagement(unittest.TestCase):
    def test_scores_in_range_and_attention_matters(self):
        attentive = make_records("Person 1", 0, 60, facing=True)
        away = make_records("Person 2", 0, 60, facing=False, motion=0.0)
        audio = np.ones(60, bool)
        w1, _ = compute_engagement(attentive, 60, audio, 30)
        w2, _ = compute_engagement(away, 60, audio, 30)
        for w in (w1, w2):
            self.assertTrue(((w["engagement"] >= 0) & (w["engagement"] <= 100)).all())
        self.assertGreater(w1["engagement"].mean(), w2["engagement"].mean())

    def test_works_without_audio(self):
        w, _ = compute_engagement(make_records("Person 1", 0, 60), 60, None, 30)
        self.assertTrue(w["audio"].isna().all())

    def test_student_analytics_columns(self):
        recs = make_records("Person 1", 0, 60)
        w, pw = compute_engagement(recs, 60, None, 30)
        s = student_analytics(recs, pw, 0.25, None)
        self.assertIn("Avg engagement", s.columns)
        self.assertEqual(len(s), 1)


class TestMLModel(unittest.TestCase):
    def test_model_prefers_engaged_features(self):
        model, acc = train_model(save=False)
        self.assertGreater(acc, 0.8)
        high = predict_engagement(model, 0.95, 0.5, 0.6)
        low = predict_engagement(model, 0.05, 0.0, 0.0)
        self.assertGreater(high, low)
        self.assertTrue(0 <= low <= 100 and 0 <= high <= 100)


class TestSuggestions(unittest.TestCase):
    def test_drop_is_detected(self):
        eng = [80, 80, 30, 25, 80, 80]
        w = pd.DataFrame({"label": [f"{i:02d}:00-{i:02d}:30" for i in range(6)],
                          "engagement": eng, "people": [4] * 6, "audio": [30] * 6})
        tips = generate_suggestions(w, pd.DataFrame())
        self.assertTrue(any("Attention dropped" in t for _, t in tips))

    def test_empty_data_is_handled(self):
        tips = generate_suggestions(pd.DataFrame(), pd.DataFrame())
        self.assertEqual(tips[0][0], "info")


if __name__ == "__main__":
    unittest.main()
