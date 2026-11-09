# 3DRespiView Trainer

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26n--OBB-111F68)
![Label Studio](https://img.shields.io/badge/Label%20Studio-annotation-FF7557)
![TFLite](https://img.shields.io/badge/export-TFLite-FF6F00?logo=tensorflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![Platforms](https://img.shields.io/badge/runs%20on-macOS%20%7C%20Linux-lightgrey)

Annotation and training pipeline of the YOLO OBB model that places the thoraco-abdominal region of
interest in the 3DRespiView Android app. Label Studio is used to annotate IR frames, Ultralytics to
train a YOLO26n-OBB (oriented bounding boxes) model, which is exported to TFLite for the app.

Everything runs in Docker: no Python, PyTorch or CUDA to install on the machine.

## Workspace layout

This repository only holds code. The data lives next to it, in the `3DRespiView-Results` folder shared
with the [viewer](https://github.com/sadc-lab/3DRespiView_Mobile-server), with the same layout on the
Mac and on the hospital tower:

```
~/Documents/POLYMTL/
├── 3DRespiView-Viewer/                dashboard + workstation scripts (pull, sync, doctor)
├── 3DRespiView-Trainer/               this repository
└── 3DRespiView-Results/
    ├── sessions/                      sessions recorded by the phone
    └── training/
        ├── 1_raw_data/                IR frames to annotate (served to Label Studio)
        ├── 2_label_studio/            Label Studio database and uploads
        ├── 3_yolo_dataset/<version>/  datasets exported from Label Studio
        ├── 4_training_results/<version>/   Ultralytics runs (curves, weights/best.pt)
        ├── 5_exported_models/<version>/    model_torse_<version>.tflite + model_card.json
        └── weights/                   base weights (yolo26n-obb.pt)
```

To work on another machine, copy `3DRespiView-Results` with the viewer's
`scripts/sync_results.py` (external drive). **Stop Label Studio before synchronizing**: its database
must not change during the copy, and annotating on two machines between two synchronizations
creates a conflict the script reports instead of overwriting.

## Setup

1. Install [Docker](https://www.docker.com/products/docker-desktop/) and git.
2. Clone this repository next to `3DRespiView-Results` (see the viewer's README for the full
   workstation setup), then get the training data:
   - **from another machine**: `python3 ../3DRespiView-Viewer/scripts/sync_results.py /Volumes/T7/3DRespiView-Results --from`
   - **from the former `3DRespiview-YOLO` folder** (one time): see [Migration](#migration-from-3drespiview-yolo)
   - **from scratch**: `mkdir -p ../3DRespiView-Results/training/{1_raw_data,2_label_studio}`
3. Machine-specific settings, only where needed: `cp .env.example .env`, then

   | Machine | `.env` |
   |---|---|
   | Mac (Apple Silicon) | nothing, no `.env` needed |
   | Linux with an NVIDIA GPU | `YOLO_IMAGE=ultralytics/ultralytics:latest`, `COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml` |
   | Linux without GPU | `YOLO_IMAGE=ultralytics/ultralytics:latest-cpu` |
   | Linux (both cases) | `TRAINER_USER` and `LABEL_STUDIO_USER` set to your ids (`id -u`, `id -g`) so the files stay yours |

   Use the same `LABEL_STUDIO_VERSION` on every machine (`docker exec label-studio label-studio version`).
4. `docker compose up -d`

| Service | Where |
|---|---|
| Label Studio | http://localhost:8080 |
| Training container | `yolo-training` (`docker exec -it yolo-training …`) |

## Workflow

### 1. Collect IR frames
Put the IR frames to annotate in `training/1_raw_data/` (JPEG). In Label Studio they are served as
local files: *Settings → Cloud Storage → Add Source Storage → Local files*, path `/label-studio/images`.

### 2. Annotate in Label Studio
Class `thoracoabdominal`, outlined with 4 points (*PolygonLabels*, or *RectangleLabels* with rotation)
to produce an oriented box.

### 3. Export a dataset version
*Export → YOLO* in Label Studio, then extract the archive flat into a new version folder:

```
training/3_yolo_dataset/v4_obb/images/*.jpg
training/3_yolo_dataset/v4_obb/labels/*.txt
```

One folder per version: never add images to a version that has already been trained.

### 4. Train
```bash
docker exec -it yolo-training python scripts/pipeline.py v4_obb             # train only
docker exec -it yolo-training python scripts/pipeline.py v4_obb --tflite    # train + export for the app
```
Options: `--epochs` (100), `--imgsz` (640), `--seed` (42), `--val-ratio` (0.2).

- **Split**: 80 % train / 20 % val, reproducible with `--seed` and recorded in `split.json`. Running
  the pipeline again on the same version reuses that split.
- **Training**: curves, confusion matrices and `weights/best.pt` in `4_training_results/v4_obb/`.
- **Export** (`--tflite`): `5_exported_models/v4_obb/model_torse_v4_obb.tflite` and `model_card.json`
  (dataset version and split, training settings, metrics, Ultralytics version, **SHA-256**).

### 5. Deploy in the app
Copy the `.tflite` into the app's assets. Each analysis recorded by the app stores the model's SHA-256
in `analysis.json`: compare it with `model_card.json` to know which dataset and training run produced
the results shown in the dashboard.

## Migration from 3DRespiview-YOLO

The data used to live inside the former local folder `3DRespiview-YOLO`. To move it, one time:

```bash
cd ~/Documents/POLYMTL/3DRespiview-YOLO && docker compose down       # stop the former Label Studio
cd ~/Documents/POLYMTL/3DRespiView-Trainer
python3 scripts/migrate_legacy.py ../3DRespiview-YOLO
docker compose up -d                                                 # check the projects in Label Studio
```

The script copies `1_raw_data`, `2_label_studio`, `3_yolo_dataset`, `4_training_results`,
`5_exported_models`, `prediction_test` and `yolo26n-obb.pt` into `3DRespiView-Results/training/`,
checks every file, and never modifies the former folder. Delete it once Label Studio shows your
projects and annotations.

## Tests

```bash
python3 -m unittest discover -s tests      # pipeline (Ultralytics replaced by a stand-in) and migration
```
