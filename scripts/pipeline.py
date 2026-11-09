"""
YOLO OBB training pipeline: train/val split, training, optional TFLite export for the app.

Runs inside the training container (data mounted at /data):

    docker exec -it yolo-training python scripts/pipeline.py v4_obb
    docker exec -it yolo-training python scripts/pipeline.py v4_obb --tflite
    docker exec -it yolo-training python scripts/pipeline.py v4_obb --epochs 150 --seed 7

Dataset   /data/3_yolo_dataset/<version>/images/*.jpg + labels/*.txt (Label Studio YOLO export)
Results   /data/4_training_results/<version>/            (Ultralytics run: curves, weights/best.pt)
Model     /data/5_exported_models/<version>/model_torse_<version>.tflite + model_card.json

- The split is reproducible (--seed) and recorded in split.json. Once a dataset is split,
  running the pipeline again reuses the same split instead of reshuffling.
- model_card.json records the model's SHA-256. The app writes the same hash in every
  analysis.json, which links each analysis to its model, dataset and training run.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from datetime import datetime

DATA = os.environ.get("TRAINING_DATA_DIR", "/data")
CLASS_NAMES = {0: "thoracoabdominal"}
BASE_MODEL = "yolo26n-obb.pt"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_for(image_name):
    return os.path.splitext(image_name)[0] + ".txt"


def split_dataset(dataset_dir, seed, val_ratio):
    """Splits images/ and labels/ into train/ and val/ once; later runs reuse the split."""
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    split_file = os.path.join(dataset_dir, "split.json")

    flat = sorted(f for f in os.listdir(images_dir)
                  if f.lower().endswith(IMAGE_EXTENSIONS) and os.path.isfile(os.path.join(images_dir, f)))
    already = {s: sorted(os.listdir(os.path.join(images_dir, s)))
               for s in ("train", "val") if os.path.isdir(os.path.join(images_dir, s))}

    if already.get("train") and not flat:
        print(f"[+] Split already done, reused: {len(already['train'])} train / {len(already.get('val', []))} val")
        return already["train"], already.get("val", [])
    if already.get("train") and flat:
        sys.exit(f"[-] {images_dir} holds both split folders and {len(flat)} unsplit images. "
                 "Put the new images in a new dataset version instead.")
    if not flat:
        sys.exit(f"[-] No image found in {images_dir}. Extract the Label Studio YOLO export there "
                 "(images/ and labels/ side by side, no subfolder).")

    missing = [f for f in flat if not os.path.isfile(os.path.join(labels_dir, label_for(f)))]
    if missing:
        print(f"[!] {len(missing)} image(s) without label (kept as background), e.g. {missing[0]}")

    shuffled = flat[:]
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    val, train = sorted(shuffled[:n_val]), sorted(shuffled[n_val:])

    for split, names in (("train", train), ("val", val)):
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
        for name in names:
            shutil.move(os.path.join(images_dir, name), os.path.join(images_dir, split, name))
            label = os.path.join(labels_dir, label_for(name))
            if os.path.exists(label):
                shutil.move(label, os.path.join(labels_dir, split, label_for(name)))

    with open(split_file, "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "valRatio": val_ratio, "createdAt": datetime.now().isoformat(timespec="seconds"),
                   "train": train, "val": val}, f, indent=2)
    print(f"[+] Split (seed {seed}): {len(train)} train / {len(val)} val, recorded in split.json")
    return train, val


def write_dataset_yaml(dataset_dir):
    path = os.path.join(dataset_dir, "dataset.yaml")
    names = "\n".join(f"  {k}: {v}" for k, v in CLASS_NAMES.items())
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"path: {dataset_dir}\ntrain: images/train\nval: images/val\n\nnames:\n{names}\n")
    return path


def base_model_path():
    local = os.path.join(DATA, "weights", BASE_MODEL)
    return local if os.path.isfile(local) else BASE_MODEL  # Ultralytics downloads it if needed


def metrics_of(trainer):
    try:
        return {k: round(float(v), 4) for k, v in trainer.metrics.items()}
    except Exception:  # noqa: BLE001 - metrics are informative only
        return {}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Split, train and optionally export a YOLO OBB model.")
    parser.add_argument("version", help="dataset version, e.g. v4_obb (folder in 3_yolo_dataset/)")
    parser.add_argument("--tflite", action="store_true", help="export model_torse_<version>.tflite for the app")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42, help="seed of the train/val split and of the training")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args(argv)

    from ultralytics import YOLO
    import ultralytics

    dataset_dir = os.path.join(DATA, "3_yolo_dataset", args.version)
    if not os.path.isdir(os.path.join(dataset_dir, "images")):
        sys.exit(f"[-] Dataset not found: {dataset_dir}/images")

    train, val = split_dataset(dataset_dir, args.seed, args.val_ratio)
    yaml_path = write_dataset_yaml(dataset_dir)

    print(f"--- YOLO OBB TRAINING: {args.version} ({args.epochs} epochs, ultralytics {ultralytics.__version__}) ---")
    base = base_model_path()
    model = YOLO(base)
    model.train(data=yaml_path, epochs=args.epochs, imgsz=args.imgsz, seed=args.seed,
                project=os.path.join(DATA, "4_training_results"), name=args.version)
    run_dir = str(model.trainer.save_dir)
    best = os.path.join(run_dir, "weights", "best.pt")
    print(f"[+] Training done: {run_dir}")

    if not args.tflite:
        print("[+] Mobile export skipped (add --tflite to build the model for the app).")
        return 0

    print("--- TFLITE EXPORT ---")
    exported = str(YOLO(best).export(format="tflite", imgsz=args.imgsz) or "")
    if not os.path.isfile(exported):  # older Ultralytics: look where the export usually lands
        candidate = os.path.join(run_dir, "weights", "best_saved_model", "best_float32.tflite")
        exported = candidate if os.path.isfile(candidate) else ""
    if not exported:
        sys.exit("[-] The TFLite file could not be found after the export.")

    export_dir = os.path.join(DATA, "5_exported_models", args.version)
    os.makedirs(export_dir, exist_ok=True)
    model_name = f"model_torse_{args.version}.tflite"
    target = os.path.join(export_dir, model_name)
    shutil.copy2(exported, target)

    card = {
        "name": model_name,
        "sha256": sha256_of(target),
        "createdAt": datetime.now().isoformat(timespec="seconds"),
        "dataset": {"version": args.version, "train": len(train), "val": len(val),
                    "seed": args.seed, "valRatio": args.val_ratio},
        "training": {"baseModel": os.path.basename(base), "epochs": args.epochs, "imgsz": args.imgsz,
                     "runDir": os.path.relpath(run_dir, DATA), "bestPtSha256": sha256_of(best),
                     "ultralytics": ultralytics.__version__, "metrics": metrics_of(model.trainer)},
        "classes": CLASS_NAMES,
    }
    with open(os.path.join(export_dir, "model_card.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    print(f"[+] Model for the app: {target}")
    print(f"    sha256 {card['sha256']} (written in model_card.json; the app records it in analysis.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
