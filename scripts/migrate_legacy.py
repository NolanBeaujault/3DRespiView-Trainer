#!/usr/bin/env python3
"""
One-time migration of the data of the former training folder (3DRespiview-YOLO, where the data
lived inside the repository) into ../3DRespiView-Results/training/.

    python3 scripts/migrate_legacy.py ~/Documents/POLYMTL/3DRespiview-YOLO

Copies (never moves, never deletes): 1_raw_data, 2_label_studio, 3_yolo_dataset,
4_training_results, 5_exported_models, prediction_test, and yolo26n-obb.pt into weights/.
Files already present with the same size are skipped; a file that differs is reported and
left untouched. The former folder stays as it is: keep it as a backup until the new setup
has been checked, then delete it.

Stop Label Studio in the former folder first (docker compose down there).
Standard library only.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRAINING = os.path.join(os.path.dirname(REPO), "3DRespiView-Results", "training")

FOLDERS = ["1_raw_data", "2_label_studio", "3_yolo_dataset", "4_training_results", "5_exported_models",
           "prediction_test"]
FILES = {"yolo26n-obb.pt": "weights/yolo26n-obb.pt"}
SKIPPED = {".DS_Store", "Thumbs.db"}


def skipped(name: str) -> bool:
    return name in SKIPPED or name.startswith("._") or name.endswith(("-wal", "-shm"))


def plan(src_root: str, dst_root: str) -> list:
    """[(src, dst)] of every file to consider."""
    pairs = []
    for folder in FOLDERS:
        src_dir = os.path.join(src_root, folder)
        if not os.path.isdir(src_dir):
            continue
        for current, dirs, files in os.walk(src_dir):
            dirs[:] = [d for d in dirs if not skipped(d)]
            for name in files:
                if skipped(name):
                    continue
                src = os.path.join(current, name)
                pairs.append((src, os.path.join(dst_root, os.path.relpath(src, src_root))))
    for name, target in FILES.items():
        src = os.path.join(src_root, name)
        if os.path.isfile(src):
            pairs.append((src, os.path.join(dst_root, target)))
    return pairs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Copie les données de l'ancien dossier 3DRespiview-YOLO "
                                                 "vers 3DRespiView-Results/training.")
    parser.add_argument("legacy", help="ancien dossier, ex. ~/Documents/POLYMTL/3DRespiview-YOLO")
    parser.add_argument("--training", default=DEFAULT_TRAINING, help=f"destination (défaut : {DEFAULT_TRAINING})")
    args = parser.parse_args(argv)

    src_root = os.path.abspath(os.path.expanduser(args.legacy))
    dst_root = os.path.abspath(os.path.expanduser(args.training))
    if not any(os.path.isdir(os.path.join(src_root, f)) for f in FOLDERS):
        print(f"Erreur : {src_root} ne contient aucun des dossiers {', '.join(FOLDERS)}.", file=sys.stderr)
        return 1
    if os.path.commonpath([src_root, dst_root]) == src_root:
        print("Erreur : la destination est à l'intérieur de l'ancien dossier.", file=sys.stderr)
        return 1
    wal = os.path.join(src_root, "2_label_studio", "label_studio.sqlite3-wal")
    if os.path.isfile(wal) and os.path.getsize(wal) > 0:
        print("Erreur : Label Studio tourne encore sur l'ancien dossier. Arrête-le :\n"
              f"  cd {src_root} && docker compose down", file=sys.stderr)
        return 1

    copied = skipped_same = 0
    conflicts = []
    total_bytes = 0
    for src, dst in plan(src_root, dst_root):
        size = os.path.getsize(src)
        if os.path.exists(dst):
            if os.path.getsize(dst) == size:
                skipped_same += 1
            else:
                conflicts.append(dst)
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst + ".part")
        os.replace(dst + ".part", dst)
        copied += 1
        total_bytes += size

    for folder in ("1_raw_data", "2_label_studio", "3_yolo_dataset", "4_training_results", "5_exported_models",
                   "weights"):
        os.makedirs(os.path.join(dst_root, folder), exist_ok=True)

    # Check: every source file now exists at the destination with the same size
    missing = [dst for src, dst in plan(src_root, dst_root)
               if not os.path.isfile(dst) or os.path.getsize(dst) != os.path.getsize(src)]

    print(f"Destination : {dst_root}")
    print(f"{copied} fichier(s) copié(s) ({total_bytes / 1024 ** 2:.0f} Mo), {skipped_same} déjà présent(s).")
    if conflicts:
        print(f"{len(conflicts)} fichier(s) différent(s) déjà présent(s), non écrasé(s), ex. {conflicts[0]}")
    if missing:
        print(f"Vérification : {len(missing)} fichier(s) manquant(s) ou différent(s), ex. {missing[0]}")
        return 2
    print("Vérification : toutes les données sont présentes à l'identique.")
    print(f"\nL'ancien dossier n'a pas été modifié : garde-le jusqu'à avoir vérifié Label Studio\n"
          f"(cd {REPO} && docker compose up -d), puis supprime-le.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
