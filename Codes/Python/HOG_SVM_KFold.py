import os
import glob 
import numpy as np 
import cv2
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedKFold
import matplotlib.pyplot as plt


os.chdir('C:/Users/blaze03/Desktop/tmp')
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

win_size = (96, 48)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
num_bins = 9
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)

# =============================================================================
# X = np.zeros((sum(no_imgs),320)) # Feature Matrix
# cnt = 0
# =============================================================================
x_pos=[]
for i in range(len(list_fams)):
    os.chdir(list_fams[i])
    img_list = glob.glob('*.jpg') # Getting only 'png' files in a folder
    for j in range(len(img_list)):
        image = cv2.imread(img_list[j])
        img_hog = hog.compute(image,(64,64))
        x_pos.append(img_hog)
    os.chdir('..')
x_pos = np.array(x_pos, dtype=np.float32)
x_new01 = x_pos.reshape(x_pos.shape[0], x_pos.shape[1])
print(x_new01.shape) #with the matrix (no of images, features per image)
# =============================================================================
# cross validation 
# =============================================================================
C_p = 1.5
clf= svm.SVC(kernel='linear', C= C_p)

conf_mat = np.zeros((len(no_imgs),len(no_imgs)))
rkf = RepeatedKFold(n_splits = 10, n_repeats = 10)
for train, test in rkf.split(x_new01):
    X_train, X_test, y_train, y_test = x_new01[train], x_new01[test], y[train], y[test]
    clf.fit(X_train,y_train)
    y_predict = clf.predict(X_test)
    cm = confusion_matrix(y_test,y_predict)
    conf_mat = conf_mat + cm

conf_mat = conf_mat.T # since rows and  cols are interchanged
avg_acc = np.trace(conf_mat)/(10*sum(no_imgs))
conf_mat_norm = np.array(conf_mat)/(np.array(no_imgs, dtype =int)*10) # Normalizing the confusion matrix
print('Computing Confusion Matrix')
plt.imshow(conf_mat_norm, interpolation='nearest')
plt.title('Confusion matrix')
plt.colorbar()
plt.show()
print('Accuracy of the method is', avg_acc)
