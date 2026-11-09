"""Tests of scripts/migrate_legacy.py on a fake former 3DRespiview-YOLO folder."""

import io
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))

import migrate_legacy  # noqa: E402


def write(path, data="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(data)


class MigrateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.legacy = os.path.join(self.tmp, "3DRespiview-YOLO")
        self.training = os.path.join(self.tmp, "3DRespiView-Results", "training")
        write(os.path.join(self.legacy, "1_raw_data", "IR4OBB_20260521_124006.jpg"), "jpg")
        write(os.path.join(self.legacy, "2_label_studio", "label_studio.sqlite3"), "db")
        write(os.path.join(self.legacy, "2_label_studio", "media", "upload", "1", "a.jpg"), "up")
        write(os.path.join(self.legacy, "3_yolo_dataset", "v3_obb", "images", "train", "a.jpg"), "a")
        write(os.path.join(self.legacy, "4_training_results", "v3_obb", "weights", "best.pt"), "pt")
        write(os.path.join(self.legacy, "5_exported_models", "v3_obb", "model_torse_v3.tflite"), "tfl")
        write(os.path.join(self.legacy, "prediction_test", "pred.jpg"), "p")
        write(os.path.join(self.legacy, "yolo26n-obb.pt"), "base")
        write(os.path.join(self.legacy, "scripts", "split.py"), "code stays in git")
        write(os.path.join(self.legacy, "1_raw_data", ".DS_Store"), "junk")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_main(self, *args):
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(out):
            code = migrate_legacy.main([self.legacy, "--training", self.training, *args])
        return code, out.getvalue()

    def test_copies_data_only_and_is_idempotent(self):
        code, out = self.run_main()
        self.assertEqual(code, 0, out)
        for rel in ("1_raw_data/IR4OBB_20260521_124006.jpg", "2_label_studio/label_studio.sqlite3",
                    "2_label_studio/media/upload/1/a.jpg", "3_yolo_dataset/v3_obb/images/train/a.jpg",
                    "4_training_results/v3_obb/weights/best.pt", "5_exported_models/v3_obb/model_torse_v3.tflite",
                    "prediction_test/pred.jpg", "weights/yolo26n-obb.pt"):
            self.assertTrue(os.path.isfile(os.path.join(self.training, rel)), rel)
        self.assertFalse(os.path.exists(os.path.join(self.training, "scripts")))
        self.assertFalse(os.path.exists(os.path.join(self.training, "1_raw_data", ".DS_Store")))
        self.assertTrue(os.path.isfile(os.path.join(self.legacy, "2_label_studio", "label_studio.sqlite3")))

        code, out = self.run_main()
        self.assertEqual(code, 0)
        self.assertIn("0 fichier(s) copié(s)", out)

    def test_refuses_while_label_studio_runs(self):
        write(os.path.join(self.legacy, "2_label_studio", "label_studio.sqlite3-wal"), "pending")
        code, out = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("Label Studio tourne encore", out)
        self.assertFalse(os.path.exists(self.training))

    def test_never_overwrites(self):
        write(os.path.join(self.training, "2_label_studio", "label_studio.sqlite3"), "a different database")
        code, out = self.run_main()
        self.assertEqual(code, 2)
        with open(os.path.join(self.training, "2_label_studio", "label_studio.sqlite3")) as f:
            self.assertEqual(f.read(), "a different database")


if __name__ == "__main__":
    unittest.main()
