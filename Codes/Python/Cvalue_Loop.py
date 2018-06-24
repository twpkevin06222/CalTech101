
# =============================================================================
# Libraries
# =============================================================================
import os
import glob 
import numpy as np 
import cv2
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage.feature import hog
from sklearn.model_selection import cross_val_score
# =============================================================================
# Directories and Labelling 
# =============================================================================
os.chdir('/home/blaze03/ImageUnderstanding/tmp_256')
list_fams = os.listdir(os.getcwd()) # vector of strings with family names

no_imgs = [] # No. of samples per family

for i in range(len(list_fams)):
    os.chdir(list_fams[i])
    len1 = len(glob.glob('*.jpg')) # assuming the images are stored as 'jpg'
    no_imgs.append(len1)
    os.chdir('..')

total = sum(no_imgs) # total number of all samples
y = np.zeros(total) # label vector

temp1 = np.zeros(len(no_imgs)+1)
temp1[1:len(temp1)]=no_imgs
temp2 = int(temp1[0]); # now temp2 is [0 no_imgs]

for jj in range(len(no_imgs)): 
    temp3 = temp2 +int(temp1[jj+1])
    for ii in range(temp2,temp3): 
       y[ii] = jj
    temp2 = temp2+ int(temp1[jj+1])
y = y.astype(np.int32)    
# =============================================================================
# HOG 
# =============================================================================
def fd_hog(image):
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(32, 32),
                    cells_per_block=(2, 2), visualise=True)
    
    return fd


x_pos=[]
for i in range(len(list_fams)):
    os.chdir(list_fams[i])
    img_list = glob.glob('*.jpg') # Getting only 'png' files in a folder
    for j in range(len(img_list)):
        image = cv2.imread(img_list[j])
        image= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        img_hog = fd_hog(image)
        x_pos.append(img_hog)
    os.chdir('..')
x_pos = np.array(x_pos, dtype=np.float32)
x_new01 = x_pos.reshape(x_pos.shape[0], x_pos.shape[1])
print('Total number of images, feature vector:',x_new01.shape)
print()

# =============================================================================
# PCA
# =============================================================================
pca = PCA()
pca_fit = pca.fit_transform(x_new01,y)
# =============================================================================
# C value loop
# =============================================================================
C_range = np.arange(0,11) #could be change according to desire range
C_scores = []
h = 0
x_p = pca_fit
for k in C_range:
    if k==0:
        k =1 
    clf= LinearSVC(C=k)
    scores = cross_val_score(clf, x_p, y, cv =10, scoring = 'accuracy' )
    C_scores.append(scores.mean())
    h+=1
    print(h)
    
print(C_scores)

plt.plot(C_range, C_scores)
plt.xlabel('Value of C for SVM')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
