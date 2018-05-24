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
3. To generate feature vectors of the same size for HOG, all images are resize to (128x64). 


