import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from skimage import filters
from skimage.feature import graycomatrix, graycoprops
from scipy.fftpack import fft2
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image, ImageDraw


# Constantes #
# Chemins des dossiers/fichiers
BASE_PATH = "./dataTest/"
TARGET_PATH = "target.png"
# On récupère les différentes catégories d'images à partir des noms des dossiers où elles sont stockées
CATEGORIES = [folder for folder in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, folder))]
LABEL_MAP = {category: idx for idx, category in enumerate(CATEGORIES)}


# Fonctions de descripteurs #
def extract_contour_features(image):
	"""
	Donne des caractéristiques sur l'image en fonction des contours.
	:param image: Image à traiter.
	:return: Les caractéristiques de contour.
	"""
	# Version Sobel
	edges = filters.sobel(image)
	return cv2.resize(edges, (64, 64)).flatten()  # Fixe la taille

	# Version passe-haut
	# Appliquer un filtre passe-haut
	# blurred = cv2.GaussianBlur(image, (5, 5), 0)  # Floutage pour lisser l'image
	# high_pass = cv2.subtract(image, blurred)  # Passe-haut en soustrayant l'image floutée de l'image originale

	# Redimensionner et aplatir l'image filtrée
	# return cv2.resize(high_pass, (64, 64)).flatten()


def extract_frequency_features(image):
	"""
	Donne des caractéristiques sur l'image en fonction des fréquences.
	:param image: Image à traiter.
	:return: Les caractéristiques de fréquence.
	"""
	f_transform = np.abs(fft2(image)) # Transformation de Fourier
	f_transform = np.log(f_transform + 1)  # Log pour réduire l'échelle (données concentrées en un point sans ça)
	f_transform_resized = cv2.resize(f_transform, (32, 32))
	return f_transform_resized.flatten()  # Fixe la taille


def extract_texture_features(image):
	"""
	Donne des caractéristiques sur l'image en fonction des textures.
	:param image: Image à traiter.
	:return: Les caractéristiques de texture.
	"""
	hist = cv2.calcHist([image], [0], None, [128], [0, 256]) # Histogramme HSV
	return hist.flatten()


def extract_glcm_features(image):
	"""
	Donne des caractéristiques sur l'image en fonction des matrices de coocurrences.
	:param image: Image à traiter.
	:return: Les caractéristiques GLCM.
	"""
	image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

	# Calculer la matrice GLCM
	glcm = graycomatrix(
		image,
		distances=[1],  # Distance entre les pixels
		angles=[0],  # Angles : 0°
		levels=256,  # Niveaux de gris
		symmetric=True,
		normed=True
	)

	# Extraire des propriétés GLCM (contraste, homogénéité, énergie, corrélation)
	features = []
	properties = ['contrast', 'homogeneity', 'energy', 'correlation']
	for prop in properties:
		props = graycoprops(glcm, prop)	# Récupérer les valeurs des propriétés
		features.extend(props.flatten())  # Ajouter les valeurs de chaque angle

	return np.array(features)


# Altérer image #
def alter_image(image):
	"""
	Applique des modifications sur l'image donnée
	:param image: Image à modifier
	:return: Image modifiée
	"""
	# Appliquer une rotation de 90 degrés
	rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

	# Redimensionner pour maintenir la taille d'origine sans bords noirs
	return cv2.resize(rotated_image, (image.shape[1], image.shape[0]))


# Random Forest #
def random_forest():
	"""
	Génère et fournit des données Random Forest

	:return: scaler, pca, clf, cm, report
	"""
	features = []
	labels = []
	for category in CATEGORIES:
		folder_path = os.path.join(BASE_PATH, category)
		for file_name in os.listdir(folder_path):

				# Ouverture des images 
				image_path = os.path.join(folder_path, file_name)
				image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
				if image is None:
					continue
  
				# Extraction des caractéristiques
				contour_features = extract_contour_features(image)
				frequency_features = extract_frequency_features(image)
				texture_features = extract_texture_features(image)
				glcm_features = extract_glcm_features(image)
				combined_features = np.concatenate([contour_features, frequency_features, texture_features, glcm_features])
    
				# Centralisation des données
				features.append(combined_features)
				labels.append(LABEL_MAP[category])

	# Conversion en matrice et normalisation
	X = np.array(features)
	y = np.array(labels)
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)

	
	# Réduction de dimensions
	pca = PCA(min(100, X_scaled.shape[1]))
	X_reduced = pca.fit_transform(X_scaled)

	# Random Forest
	weights = {
		LABEL_MAP['water']: 0.5
	}
	X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)
	clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=weights)
	#clf = KMeans(n_clusters=len(CATEGORIES), random_state=42)
	clf.fit(X_train, y_train)
 

	# Évaluation
	y_pred = clf.predict(X_test)
	cm = confusion_matrix(y_test, y_pred)
	report = classification_report(y_test, y_pred)
	

	return scaler, pca, clf, cm, report


# Prédiction d'une image #
def predict_image_category(image_path, scaler, pca, clf):
	"""
	Affiche la catégorie de l'image donnée.

	:param image_path: Le chemin conduisant à l'image.
	:param scaler: Scaler utilisé pour normaliser les données.
	:param pca: PCA utilisé pour réduire les dimensions.
	:param clf: Modèle de Random Forest.
	"""
 
	# Charger l'image
	image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	image = alter_image(image)

	if image is None:
		print("Erreur : l'image n'a pas pu être chargée.")
		return

	# Extraire les features
	contour_features = extract_contour_features(image)
	frequency_features = extract_frequency_features(image)
	texture_features = extract_texture_features(image)
	glcm_features = extract_glcm_features(image)
	combined_features = np.concatenate([contour_features, frequency_features, texture_features, glcm_features])

	# Prétraitement (normalisation et réduction de dimension)
	combined_features_scaled = scaler.transform([combined_features])  # Normalise les features
	combined_features_reduced = pca.transform(combined_features_scaled)  # Réduit les dimensions

	# Prédiction
	predicted_label = clf.predict(combined_features_reduced)

	# Résultat
	predicted_category = [key for key, value in LABEL_MAP.items() if value == predicted_label[0]]
	print(f"L'image est classée dans la catégorie : {predicted_category[0]}")


# Segmentation d'une image en rectangles #
def calculate_rectangle_size(image_width, image_height, num_width, num_height):
	"""
	Calcule la taille des rectangles en fonction du nombre de rectangles en largeur et hauteur.

	:param image_width: Largeur de l'image.
	:param image_height: Hauteur de l'image.
	:param num_width: Nombre de rectangles en largeur.
	:param num_height: Nombre de rectangles en hauteur.
	:return: (width, height) Taille de chaque rectangle.
	"""
	rect_width = image_width // num_width
	rect_height = image_height // num_height
	return rect_width, rect_height


def get_color_for_category(category):
	"""
	Associe une couleur à une catégorie.
	:param category: Nom de la catégorie.
	:return: Tuple de couleur RGBA.
	"""
	color_map = {
		"green_area": (0, 255, 0, 128),  # Vert semi-transparent
		"water": (0, 0, 255, 150),       # Bleu semi-transparent
		"desert": (255, 255, 0, 128),    # Jaune semi-transparent
		"cloudy": (128, 128, 128, 128)   # Gris semi-transparent
	}
	return color_map.get(category, (255, 0, 0, 128))  # Rouge par défaut


def predict_rectangle_category(image, scaler, pca, clf):
	"""
	Prédit la catégorie d'une région d'image.
	:param image: Région de l'image (un rectangle).
	:param scaler: Scaler utilisé pour normaliser les données.
	:param pca: PCA utilisé pour réduire les dimensions.
	:param clf: Modèle de Random Forest.
	:return: Nom de la catégorie prédite.
	"""
	contour_features = extract_contour_features(image)
	frequency_features = extract_frequency_features(image)
	texture_features = extract_texture_features(image)
	glcm_features = extract_glcm_features(image)
	combined_features = np.concatenate([contour_features, frequency_features, texture_features, glcm_features])

	# Normalisation et réduction
	combined_features_scaled = scaler.transform([combined_features])
	combined_features_reduced = pca.transform(combined_features_scaled)

	# Prédiction
	predicted_label = clf.predict(combined_features_reduced)
	return [key for key, value in LABEL_MAP.items() if value == predicted_label[0]][0]


def colorize_image_with_rectangles(image_path, num_width, num_height, output_path, scaler, pca, clf):
	"""
	Transforme une image en la superposant avec des rectangles colorés selon les catégories prédites.

	:param image_path: Chemin de l'image d'entrée.
	:param num_width: Nombre de rectangles en largeur.
	:param num_height: Nombre de rectangles en hauteur.
	:param output_path: Chemin de sauvegarde de l'image générée.
	:param scaler: Scaler pour normaliser les données.
	:param pca: PCA pour réduire les dimensions.
	:param clf: Modèle de Random Forest.
	"""
	# Charger l'image
	image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	if image is None:
		print("Erreur : l'image n'a pas pu être chargée.")
		return
	height, width = image.shape

	# Calculer la taille des rectangles
	rect_width, rect_height = calculate_rectangle_size(width, height, num_width, num_height)

	# Créer une image RGBA pour l'overlay
	overlay = Image.new("RGBA", (width, height))
	draw = ImageDraw.Draw(overlay)

	# Parcourir les rectangles et prédire leurs catégories
	for row in range(num_height):
		for col in range(num_width):
				x0 = col * rect_width
				y0 = row * rect_height
				x1 = min(x0 + rect_width, width)
				y1 = min(y0 + rect_height, height)

				# Extraire le rectangle
				rectangle = image[y0:y1, x0:x1]

				# Prédire la catégorie
				category = predict_rectangle_category(rectangle, scaler, pca, clf)
				color = get_color_for_category(category)

				# Dessiner le rectangle avec la couleur associée
				draw.rectangle([x0, y0, x1, y1], fill=color)

	# Fusionner l'image originale et l'overlay
	original_image = Image.open(image_path).convert("RGBA")
	output_image = Image.alpha_composite(original_image, overlay)

	# Sauvegarder l'image finale
	output_image.save(output_path, "PNG")
	print(f"Image sauvegardée sous {output_path}")


# Affichage résultat random forest #
def print_result(report):
	plt.figure(figsize=(8, 6))
	sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr", xticklabels=CATEGORIES, yticklabels=CATEGORIES)
	plt.xlabel('Prédictions')
	plt.ylabel('Réalité')
	plt.show()
	print("\nRapport de classification:\n", report)


if __name__ == '__main__':
	# Entraînement random forest
	scaler, pca, clf, cm, report = random_forest()

	# Prédiction de l'image voulue
	# predict_image_category(TARGET_PATH, scaler, pca, clf)

	colorize_image_with_rectangles(TARGET_PATH, 20, 20, "result.png", scaler, pca, clf)

	# Affichage des résultats
	print_result(report)

	# Affichage des résultats
	print("\nMatrice de confusion:\n", confusion_matrix)
	print("\nRapport de classification:\n", report)
