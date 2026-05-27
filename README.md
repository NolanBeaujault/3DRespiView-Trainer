# 3DRespiView-Trainer : Pipeline d'annotation & d'entraînement YOLO OBB

Ce dépôt contient l'infrastructure logicielle complète pour gérer l'annotation web, l'entraînement de réseaux de neurones YOLOv26n-OBB (Oriented Bounding Boxes) et la compilation mobile pour le projet de suivi respiratoire 3DRespiView.

## Prérequis
L'ensemble de l'environnement est isolé dans des conteneurs Docker. Aucune dépendance locale (Python, PyTorch, pilotes Nvidia CUDA) n'est requise.
* Installez [Docker Desktop](https://www.docker.com/products/docker-desktop/)
* Installez [Git](https://git-scm.com/)

---

## Architecture des Fichiers du Projet

```text
.
├── 1_label_studio/           # Base de données et persistance de Label Studio (Géré par l'outil)
│   └── media/                # Stockage local sécurisé des images téléversées par le web
├── 2_yolo_dataset/           # Jeux de données versionnés extraits de Label Studio
│   └── v4_obb/               # Exemple de version (contenant dossiers /images et /labels)
├── 3_training_results/       # Résultats d'entraînement (Courbes de perte, matrices, best.pt)
├── 4_exported_models/        # Livrables compressés au format .tflite pour Android
├── scripts/
│   └── pipeline.py           # Script d'automatisation : Split Train/Val + Training (+ Export Mobile)
├── docker-compose.yml        # Orchestration Docker (Label Studio + YOLO Container)
└── yolo26n-obb.pt            # Poids de base pour le Transfer Learning

```

---

## Guide d'Utilisation Clinique

### Étape 1 : Initialisation de l'infrastructure

Démarrez les services d'arrière-plan en exécutant cette commande depuis la racine de ce dossier :

```bash
docker compose up -d

```

* **Interface graphique de labellisation :** Accessible dans votre navigateur sur `http://localhost:8080`.
* **Moteur d'entraînement YOLO :** S'exécute en tâche de fond sous le nom de conteneur `yolo-training`.

### Étape 2 : Annotation en ligne (Label Studio)

1. Ouvrez `http://localhost:8080`, créez un compte local et initialisez un nouveau projet.
2. **Importation :** Cliquez sur le bouton **Import** et glissez-déposez directement vos clichés de patients au format `.jpg` dans le navigateur.
*(Note : Les images sont automatiquement stockées de manière permanente sur votre disque dur dans le dossier `1_label_studio/media/upload/`, aucun risque de perte à l'arrêt du PC).*
3. **Configuration du projet :** Dans le menu *Labeling Setup*, sélectionnez l'outil **PolygonLabels** (ou *RectangleLabels* en activant la rotation) et ajoutez le nom de classe exact : `thoracoabdominal`.
4. **Annotation :** Détourez finement la zone thoracique à l'aide de 4 points pour créer la boîte orientée (OBB).

### Étape 3 : Exportation & Extraction du jeu de données

1. Une fois les images annotées, cliquez sur le bouton **Export** en haut à droite de l'interface web de Label Studio.
2. Choisissez impérativement le format **YOLO** (Label Studio convertit automatiquement vos polygones graphiques en vecteurs OBB natifs).
3. Téléchargez l'archive `.zip` générée.
4. Créez un dossier de version dans votre espace local (ex: `2_yolo_dataset/v4_obb/`).
5. Extrayez l'archive téléchargée pour organiser les fichiers à plat comme suit :
* Déposez l'intégralité des fichiers texte de coordonnées dans : `2_yolo_dataset/v4_obb/labels/`
* Déposez l'intégralité des images correspondantes dans : `2_yolo_dataset/v4_obb/images/`



*(Note : Ne créez pas de sous-dossiers manuellement. Laissez les images et les labels en vrac dans leurs répertoires respectifs. La pipeline se chargera de la répartition de manière autonome).*

### Étape 4 : Exécution automatique de la pipeline YOLO

Exécutez le script d'orchestration maître directement à l'intérieur de l'unité de calcul YOLO selon vos besoins de recherche :

#### Option A : Entraînement Standard (Recherche & Analyse PC)

Pour lancer l'entraînement, évaluer les métriques et générer le fichier de poids PyTorch (`best.pt`) sans compilation mobile :

```bash
docker exec -it yolo-training python scripts/pipeline.py v4_obb

```

#### Option B : Entraînement Complet avec Export Mobile (Déploiement Android)

Si vous souhaitez compiler, quantifier et exporter automatiquement le modèle finalisé pour l'application mobile, ajoutez le drapeau `--tflite` à la fin de la commande :

```bash
docker exec -it yolo-training python scripts/pipeline.py v4_obb --tflite

```

---

## Livrables Générés

La pipeline gère l'ensemble des tâches de manière transparente :

1. **Split Mathématique :** Isole aléatoirement 80% des clichés pour l'entraînement (`train/`) et 20% pour la validation clinique (`val/`).
2. **Métrologie de convergence :** Exécute 100 époques d'ajustement et écrit les courbes de précision, de rappel et les matrices de confusion dans `3_training_results/v4_obb/`.
3. **Livrable Mobile (Si option activée) :** Compile le réseau de neurones et livre le fichier compressé dans `4_exported_models/v4_obb/model_torse_v4_obb.tflite`, prêt à être copié-collé dans le répertoire d'assets de l'application Android.

```
