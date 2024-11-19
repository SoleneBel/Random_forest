 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger l'image en niveaux de gris
image_path = "./Images/Lacs/lac_d_aiguebellette.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
	print("Erreur : l'image n'a pas pu être chargée.")
else:
	# Appliquer une rotation de 90 degrés
	rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

	# Redimensionner pour maintenir la taille d'origine sans bords noirs
	rotated_resized = cv2.resize(rotated_image, (image.shape[1], image.shape[0]))

	plt.subplot(1, 2, 1)
	plt.title("Image originale")
	plt.imshow(image, cmap='gray')
	plt.axis('off')

	plt.subplot(1, 2, 2)
	plt.title("Spectre de Fourier")
	plt.imshow(rotated_image, cmap='gray')
	plt.axis('off')

	plt.tight_layout()
	plt.show()
