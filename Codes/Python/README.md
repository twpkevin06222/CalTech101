#Disclaimer: 
These codes are not for professional programmers or computer scientist. The codes are written with the intention of fast prototyping and
minimal complexity. Computational performance might be compromised. These codes are best suitable for those whom want to have fun with 
image understanding and passion for computer vision. 

Criterion:
* Feature descriptor: HOG
* Classifier: SVM (Linear Kernel)
* Validation: K-fold Cross Validation ( K =10) 

Data Processing:
1. HOG_data_preprocessing.py, handles the preprocessing of the images from CalTech101. 
2. For computational performance all images are converted to gray scale. 
3. To generate feature vectors of the same size for HOG, all images are resize to (256x256). 
4. Background_google class is removed.*(Personal preferance)  


HOG Parameters:
hog_image = hog(image, orientations=9, pixels_per_cell=(32, 32),cells_per_block=(2, 2), visualise=True)

Total number of images, feature vector: (8677, 1764)

Accuracy of the method is 0.534099276703

Computation time: 1168.012040 s

