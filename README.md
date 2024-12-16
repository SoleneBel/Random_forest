# Random Forest Image Classifier

## Description
Ce projet utilise un algorithme de classification Random Forest pour classer des images en fonction de cinq catégories prédéfinies :
- **Cloudy**
- **Desert**
- **Green Area**
- **Urban**
- **Water**

En plus de la classification d'image, le projet inclut un outil de visualisation permettant d'analyser les effets des caractéristiques de l'image, telles que les contours, les textures, les fréquences et les propriétés des matrices de co-occurrence de niveaux de gris (GLCM).

## Fonctionnalités

### 1. **Classification des images par Random Forest**
- Extraction de plusieurs types de caractéristiques d'image :
  - **Contours** (filtre Sobel)
  - **Fréquences** (Transformée de Fourier)
  - **Textures** (histogrammes)
  - **Propriétés GLCM** (co-occurrence des pixels)
- Normalisation des données via StandardScaler
- Réduction de la dimension via PCA (Analyse en Composantes Principales)
- Entraînement d’un classificateur Random Forest
- Prédiction de la catégorie des images en fonction des caractéristiques extraites
- Visualisation des résultats via une matrice de confusion et un rapport de classification

### 2. **Visualisation des effets des caractéristiques**
- Analyse des **propriétés GLCM** (Contraste, homogénéité, énergie et corrélation) appliquées à une image.
- Affichage des résultats de l’analyse des GLCM sous forme de graphiques et d'images visualisées à l'aide de la bibliothèque **Matplotlib**.

## Dépendances
Pour exécuter ce projet, les paquets suivants doivent être installés :

- **Python 3.7+**
- **Numpy**
- **OpenCV**
- **Scikit-learn**
- **Matplotlib**
- **Seaborn**
- **Scikit-image**
- **Pillow**

## Organisation des fichiers

```
.
├── main.py                  # Script principal pour l’entraînement et la classification des images
├── image_alteree.py         # Script de visualisation de caractéristiques pouvant être utilisée dans le script principal
├── /dataTest/               # Dossier contenant les sous-dossiers d'images pour chaque catégorie
├── /Images/                 # Dossier contenant quelques images plus détaillées extraites à la main
├── target.png               # Image cible à classer
├── result.png               # Résultat de la superposition des catégories prédites sur l’image cible
└── README.md                # Ce fichier
```

## Instructions d'utilisation

### 1. **Classification des images**
1. Placez vos images à classifier dans des sous-dossiers de `/dataTest/` (un sous-dossier par catégorie).
2. Exécutez le fichier `main.py` :
```bash
python main.py
```
3. Le programme extraira les caractéristiques des images, effectuera l’entraînement du modèle Random Forest et affichera les résultats sous forme de matrice de confusion et de rapport de classification.
4. Le programme colorisera l'image `target.png` et sauvegardera le résultat sous le nom `result.png`.
5. Il est également possible de classer plus généralement une image dans une des classes en décommentant la ligne 333. 

### 2. **Visualisation des caractéristiques de l’image**
1. Récupérez le chemin d'une image qui servira de support.
2. Modifiez le chemin de l'image dans `image_alteree.py` (variable `image_path`).
3. Exécutez le script :
```bash
python image_alteree.py
```
4. Le script affichera les propriétés GLCM de l’image et leur impact visuel. Vous pouvez remplacer GLCM par le descripteur de votre choix pour visualiser son impact.

## Exemple de résultat
- **Matrice de confusion** montrant les performances du classificateur Random Forest.
- **Superposition des catégories** sur l'image cible avec des couleurs distinctes pour chaque catégorie prédite.

## Auteurs
Ce projet a été réalisé par Solène Bellissard et Ilirian Rexhepi dans le cadre du cours OSEC501.
