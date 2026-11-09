"""
Tests of scripts/pipeline.py with a stand-in for Ultralytics (no GPU, no download).

    python3 -m unittest discover -s tests -v
"""

import json
import os
import shutil
import sys
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))


def install_fake_ultralytics(calls):
    module = types.ModuleType("ultralytics")
    module.__version__ = "0.0-test"

    class FakeTrainer:
        def __init__(self, save_dir):
            self.save_dir = save_dir
            self.metrics = {"metrics/mAP50-95(B)": 0.81234}

    class YOLO:
        def __init__(self, weights):
            self.weights = weights
            calls.append(("load", weights))

        def train(self, data, epochs, imgsz, seed, project, name):
            calls.append(("train", data, epochs, seed))
            save_dir = os.path.join(project, name)
            os.makedirs(os.path.join(save_dir, "weights"), exist_ok=True)
            with open(os.path.join(save_dir, "weights", "best.pt"), "wb") as f:
                f.write(b"weights")
            self.trainer = FakeTrainer(save_dir)

        def export(self, format, imgsz):
            out = os.path.join(os.path.dirname(self.weights), "best_saved_model", "best_float32.tflite")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "wb") as f:
                f.write(b"tflite model")
            return out

    module.YOLO = YOLO
    sys.modules["ultralytics"] = module


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.data = tempfile.mkdtemp()
        os.environ["TRAINING_DATA_DIR"] = self.data
        sys.modules.pop("pipeline", None)
        import pipeline
        self.pipeline = pipeline
        self.calls = []
        install_fake_ultralytics(self.calls)
        self.dataset = os.path.join(self.data, "3_yolo_dataset", "v4_obb")
        os.makedirs(os.path.join(self.dataset, "images"))
        os.makedirs(os.path.join(self.dataset, "labels"))
        for i in range(10):
            open(os.path.join(self.dataset, "images", f"IR4OBB_{i}.jpg"), "w").close()
            if i != 9:
                with open(os.path.join(self.dataset, "labels", f"IR4OBB_{i}.txt"), "w") as f:
                    f.write("0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n")

    def tearDown(self):
        shutil.rmtree(self.data)

    def test_train_and_export(self):
        self.assertEqual(self.pipeline.main(["v4_obb", "--tflite"]), 0)
        split = json.load(open(os.path.join(self.dataset, "split.json")))
        self.assertEqual((len(split["train"]), len(split["val"])), (8, 2))
        self.assertEqual(len(os.listdir(os.path.join(self.dataset, "labels", "train")))
                         + len(os.listdir(os.path.join(self.dataset, "labels", "val"))), 9)
        yaml = open(os.path.join(self.dataset, "dataset.yaml")).read()
        self.assertIn(f"path: {self.dataset}", yaml)
        self.assertIn("0: thoracoabdominal", yaml)

        export = os.path.join(self.data, "5_exported_models", "v4_obb")
        card = json.load(open(os.path.join(export, "model_card.json")))
        self.assertTrue(os.path.isfile(os.path.join(export, "model_torse_v4_obb.tflite")))
        self.assertEqual(card["sha256"], self.pipeline.sha256_of(os.path.join(export, "model_torse_v4_obb.tflite")))
        self.assertEqual(card["dataset"]["train"], 8)
        self.assertEqual(card["training"]["runDir"], "4_training_results/v4_obb")
        self.assertEqual(card["training"]["metrics"], {"metrics/mAP50-95(B)": 0.8123})

    def test_split_is_reproducible_and_reused(self):
        self.pipeline.main(["v4_obb"])
        first = json.load(open(os.path.join(self.dataset, "split.json")))
        self.pipeline.main(["v4_obb"])  # second run: same split, nothing moved again
        self.assertEqual(json.load(open(os.path.join(self.dataset, "split.json"))), first)
        self.assertEqual(sorted(os.listdir(os.path.join(self.dataset, "images", "val"))), first["val"])

        other = tempfile.mkdtemp()
        try:
            for name in ("images", "labels"):
                os.makedirs(os.path.join(other, name))
            for name in first["train"] + first["val"]:
                open(os.path.join(other, "images", name), "w").close()
            train, val = self.pipeline.split_dataset(other, 42, 0.2)
            self.assertEqual(val, first["val"])  # same seed, same split
        finally:
            shutil.rmtree(other)

    def test_local_base_weights_are_used(self):
        os.makedirs(os.path.join(self.data, "weights"))
        open(os.path.join(self.data, "weights", "yolo26n-obb.pt"), "w").close()
        self.pipeline.main(["v4_obb"])
        self.assertEqual(self.calls[0], ("load", os.path.join(self.data, "weights", "yolo26n-obb.pt")))

    def test_missing_dataset(self):
        with self.assertRaises(SystemExit):
            self.pipeline.main(["v9_obb"])


if __name__ == "__main__":
    unittest.main()
