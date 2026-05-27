# 3DRespiView-Trainer : Pipeline d'Annotation & d'Entraînement YOLO OBB

Ce dépôt contient la pipeline complète et automatisée pour annoter, entraîner et exporter des modèles YOLOv26n-OBB (Oriented Bounding Boxes) destinés au projet d'analyse respiratoire 3DRespiView.

## Prérequis
Toute la pipeline est entièrement conteneurisée. Aucune installation locale de Python, PyTorch ou de pilotes CUDA n'est nécessaire.
* Installer [Docker Desktop](https://www.docker.com/products/docker-desktop/)
* Installer [Git](https://git-scm.com/)

---

## Guide d'Utilisation Rapide

### Étape 1 : Lancement des conteneurs
Clonez ce dépôt sur votre machine et lancez les services en tâche de fond :
```bash
git clone git@github.com:NolanBeaujault/3DRespiView-Trainer.git
cd 3DRespiView-Trainer
docker compose up -d

```

* **Interface Label Studio (Annotation) :** Accessible sur `http://localhost:8080`
* **Unité de calcul YOLO :** S'exécute silencieusement sous le nom de conteneur `yolo-training`.

### Étape 2 : Annotation des données

1. Déposez vos images de patients au format `.jpg` brutes dans le dossier local `1_raw_data/`.
2. Ouvrez `http://localhost:8080` dans votre navigateur, créez un projet sur Label Studio et configurez une interface d'étiquetage avec des outils de type **Polygon** ou **Rectangle Orienté (OBB)**.
3. Une fois l'annotation terminée, exportez vos données depuis l'interface au format **YOLO**.

### Étape 3 : Exécution de la pipeline d'entraînement

1. Créez le dossier de votre nouvelle version dans le dépôt du jeu de données, par exemple : `3_yolo_dataset/v1_obb/`
2. Placez vos images exportées dans `3_yolo_dataset/v&_obb/images/` et vos labels textuels dans `3_yolo_dataset/v1_obb/labels/`.
3. Lancez la pipeline d'entraînement selon vos besoins de recherche :

#### Option A : Entraînement standard (Recherche & Prototypage PC)

Pour lancer l'entraînement classique et générer les courbes de performance ainsi que les poids `.pt` sans compiler le modèle pour mobile :

```bash
docker exec -it yolo-training python scripts/pipeline.py v1_obb

```

#### Option B : Entraînement complet avec export Mobile (Déploiement Android)

Si vous souhaitez compiler et exporter automatiquement le modèle finalisé pour l'application mobile, ajoutez le flag `--tflite` à la fin de la commande :

```bash
docker exec -it yolo-training python scripts/pipeline.py v1_obb --tflite

```

---

## Automatisation & Livrables du Système

Le script maître (`scripts/pipeline.py`) prend en charge l'intégralité des opérations de manière autonome :

* **Split automatique :** Répartition aléatoire et étanche des données (80% pour l'entraînement / 20% pour la validation).
* **Configuration à la volée :** Écriture automatique du fichier de description `dataset.yaml` requis par Ultralytics.
* **Résultats cliniques :** Les matrices de confusion, les courbes de perte, les scores F1 et le fichier `best.pt` sont sauvegardés dans `4_training_results/<votre_version>/`.
* **Livrable Mobile (Si flag détecté) :** Le modèle compressé et quantifié est automatiquement livré dans `5_exported_models/<votre_version>/model_torse_<votre_version>.tflite`, prêt à être injecté dans les assets de l'application Android.
