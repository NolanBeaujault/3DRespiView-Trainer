import os
import sys
import random
import shutil
from ultralytics import YOLO

if len(sys.argv) < 2:
    print("[-] Erreur : Veuillez spécifier une version (ex: python pipeline.py v4_obb)")
    sys.exit(1)

version_name = sys.argv[1]
# Vérification de la présence du flag optionnel pour le format mobile
export_tflite_requested = "--tflite" in sys.argv

base_dir = f"3_yolo_dataset/{version_name}"
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

for split in ["train", "val"]:
    os.makedirs(os.path.join(images_dir, split), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

images = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]

if len(images) == 0:
    print(f"[-] Erreur : Aucune image .jpg trouvée dans {images_dir}")
    sys.exit(1)

random.shuffle(images)
split_index = int(len(images) * 0.8)
train_images = images[:split_index]
val_images = images[split_index:]


def move_files(file_list, split_name):
    for img_name in file_list:
        txt_name = img_name.replace(".jpg", ".txt")
        shutil.move(
            os.path.join(images_dir, img_name),
            os.path.join(images_dir, split_name, img_name),
        )
        src_txt = os.path.join(labels_dir, txt_name)
        if os.path.exists(src_txt):
            shutil.move(src_txt, os.path.join(labels_dir, split_name, txt_name))


print("[+] Répartition du dataset (80% Train / 20% Val)...")
move_files(train_images, "train")
move_files(val_images, "val")

yaml_content = f"""path: /workspace/{base_dir}
train: images/train
val: images/val

names:
  0: thoracoabdominal
"""

yaml_path = os.path.join(base_dir, "dataset.yaml")
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"--- DÉMARRAGE DE L'ENTRAÎNEMENT YOLO OBB : {version_name} ---")
model = YOLO("yolo26n-obb.pt")
model.train(
    model="/workspace/yolo26n-obb.pt",
    data=f"/workspace/{yaml_path}",
    epochs=100,
    imgsz=640,
    project="/workspace/4_training_results",
    name=version_name,
)

# Phase d'exportation optionnelle conditionnée par le flag utilisateur
if export_tflite_requested:
    print("--- OPTION DETECTÉE : EXPORTATION ET QUANTIFICATION TFLITE ---")
    best_weights = f"/workspace/4_training_results/{version_name}/weights/best.pt"
    trained_model = YOLO(best_weights)
    trained_model.export(format="tflite", imgsz=640)

    export_dir = f"5_exported_models/{version_name}"
    os.makedirs(export_dir, exist_ok=True)
    tflite_src = f"/workspace/4_training_results/{version_name}/weights/best_saved_model/best_float32.tflite"

    if os.path.exists(tflite_src):
        shutil.copy(
            tflite_src, os.path.join(export_dir, f"model_torse_{version_name}.tflite")
        )
        print(
            f"[+] Livrable mobile exporté : {export_dir}/model_torse_{version_name}.tflite"
        )
else:
    print(
        "[+] Entraînement terminé avec succès. Export TFLite ignoré (aucun flag --tflite spécifié)."
    )
