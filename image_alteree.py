 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

# Charger l'image en niveaux de gris
image_path = "./Images/Lacs/lac_d_aiguebellette.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
	print("Erreur : l'image n'a pas pu être chargée.")
else:
	image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
	
	# Calculer la matrice GLCM
	glcm = graycomatrix(
		image, 
		distances=[1],  # Distance entre les pixels
		angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],  # Angles : 0°, 45°, 90°, 135°
		levels=256,  # Niveaux de gris
		symmetric=True, 
		normed=True
	)
	
	# Extraire des propriétés GLCM (contraste, homogénéité, énergie, corrélation)
	features = []
	properties = ['contrast', 'homogeneity', 'energy', 'correlation']
	for prop in properties:
		props = graycoprops(glcm, prop)
		features.extend(props.flatten())  # Ajouter les valeurs de chaque angle
	
	glcm_sum = np.sum(glcm[:, :, 0, 0], axis=0)
	glcm_image = cv2.resize(glcm_sum, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

	plt.subplot(1, 2, 1)
	plt.imshow(image, cmap='gray')
	plt.axis('off')

	plt.subplot(1, 2, 2)
	plt.imshow(glcm_image, cmap='hot')
	plt.colorbar()
	plt.axis('off')

	plt.tight_layout()
	plt.show()
