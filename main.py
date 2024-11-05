import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from skimage import filters
from scipy.fftpack import fft2
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Chemins des dossiers
base_path = "./dataTest/"
target_path = "target.jpg"
folders = ["desert", "cloudy", "green_area", "water"]
label_map = {folder: idx for idx, folder in enumerate(folders)}

# Fonctions de descripteurs
def extract_contour_features(image):
	edges = filters.sobel(image)
	return cv2.resize(edges, (64, 64)).flatten()  # Fixe la taille

def extract_frequency_features(image):
	f_transform = np.abs(fft2(image))
	f_transform = np.log(f_transform + 1)  # Log pour réduire l'échelle
	f_transform_resized = cv2.resize(f_transform, (32, 32))
	return f_transform_resized.flatten()  # Fixe la taille

def extract_texture_features(image):
	hist = cv2.calcHist([image], [0], None, [64], [0, 256])
	return hist.flatten()

# Préparation du dataset
features = []
labels = []
for folder in folders:
	folder_path = os.path.join(base_path, folder)
	for file_name in os.listdir(folder_path):
		image_path = os.path.join(folder_path, file_name)
		image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
		if image is None:
			continue
		contour_features = extract_contour_features(image)
		frequency_features = extract_frequency_features(image)
		texture_features = extract_texture_features(image)
		combined_features = np.concatenate([contour_features, frequency_features, texture_features])
		features.append(combined_features)
		labels.append(label_map[folder])

# Conversion en matrice et normalisation
X = np.array(features)
y = np.array(labels)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Réduction de dimensions
pca = PCA(n_components=100)
X_reduced = pca.fit_transform(X_scaled)

# Modèle Random Forest
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Évaluation
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Fonction de prédiction pour une nouvelle image
def predict_image_category(image_path):
	# Charger l'image
	image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	if image is None:
		print("Erreur : l'image n'a pas pu être chargée.")
		return

	# Extraire les features
	contour_features = extract_contour_features(image)
	frequency_features = extract_frequency_features(image)
	texture_features = extract_texture_features(image)
	combined_features = np.concatenate([contour_features, frequency_features, texture_features])
	
	# Prétraitement (normalisation et réduction de dimension)
	combined_features_scaled = scaler.transform([combined_features])  # Normalise les features
	combined_features_reduced = pca.transform(combined_features_scaled)  # Réduit les dimensions

	# Prédiction
	predicted_label = clf.predict(combined_features_reduced)
	
	# Afficher le résultat
	predicted_category = [key for key, value in label_map.items() if value == predicted_label[0]]
	print(f"L'image est classée dans la catégorie : {predicted_category[0]}")

# Exemple d'utilisation avec le chemin de la nouvelle image
predict_image_category(target_path)


# Affichage des résultats
print("Matrice de confusion:\n", cm)
print("\nRapport de classification:\n", report)