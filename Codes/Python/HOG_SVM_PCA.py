
# =============================================================================
# Libraries
# =============================================================================
import os
import glob 
import numpy as np 
import cv2
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import time 
from sklearn.decomposition import PCA

# =============================================================================
# Directories and Labelling 
# =============================================================================
os.chdir('/home/blaze03/ImageUnderstanding/tmp')
list_fams = os.listdir(os.getcwd()) # vector of strings with family names
list_fams = sorted(list_fams) #to ensure that the file names is sorted
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
print('Total number of images, feature vector:',x_new01.shape)
print()

# =============================================================================
# PCA
# =============================================================================
pca = PCA()
pca_fit = pca.fit_transform(x_new01,y)
# =============================================================================
# cross validation 
# =============================================================================
C_p = 1.5
clf= LinearSVC(C=C_p)
#clf= svm.SVC(kernel='rbf', gamma=0.7)
conf_mat = []
x_p = pca_fit
skf = StratifiedKFold(n_splits=10)
time_start = time.clock()
loop = 0 
for train, test in skf.split(x_p, y):
    X_train, X_test, y_train, y_test = x_p[train], x_p[test], y[train], y[test]
    clf.fit(X_train,y_train)
    y_predict = clf.predict(X_test)
    cm = confusion_matrix(y_test,y_predict)
    conf_mat.append(cm)
    loop+=1
    print('Cross-validation loop:',loop)

conf_mat = np.array(conf_mat) #convert into numpy array for convenience 
conf_mat_02 = np.zeros((len(no_imgs),len(no_imgs)))     

g = 0 #labels that do not fullfil the matrix size of confusion matrix
for n in range(len(conf_mat)):
    if(conf_mat[n].shape[0]<len(no_imgs)): #condition to check if the matrix fullfill or not
        g+=1
        continue
    conf_mat_02+=conf_mat[n]
print(g,'confusion matrix not fullfil the matrix size')
conf_mat_02 = np.array(conf_mat_02, dtype=np.float32)    
conf_mat_02 = conf_mat_02.T # since rows and  cols are interchanged
conf_mat_norm = conf_mat_02/conf_mat_02.sum(axis=1)[:,np.newaxis]
avg_acc = np.trace(conf_mat_norm)/len(list_fams)
time_elapsed = (time.clock() - time_start)

# =============================================================================
# Plot Confusion Matrix
# =============================================================================
print('Computing Confusion Matrix')
plt.imshow(conf_mat_norm, interpolation='nearest')
plt.title('Confusion matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()
plt.savefig('/home/blaze03/ImageUnderstanding/image/confusion_mat_HOG_SVM.png', dpi = 1000) #emf, eps for vector image
plt.show()
print()
# =============================================================================
# Accuracy 
# =============================================================================
print('Accuracy of the method is', avg_acc)

for n in range (len(conf_mat_norm.diagonal())):
    if(conf_mat_norm.diagonal()[n]==min(conf_mat_norm.diagonal())):
        print('The lowest accuracy will be %f for class %s' %(conf_mat_norm.diagonal()[n],
                                                              list_fams[n]))
    if(conf_mat_norm.diagonal()[n]==max(conf_mat_norm.diagonal())):
        print('The highest accuracy will be %f for class %s'%(conf_mat_norm.diagonal()[n],
                                                             list_fams[n]))
# =============================================================================
# Listing result
# =============================================================================
f = open('/home/blaze03/ImageUnderstanding/caltech101_result.txt','w+')
for n in range (len(conf_mat_norm.diagonal())):
    #print class name, accuracy per class, std dev per class
    f.write('%-20s %f %12f\n'%(list_fams[n],conf_mat_norm.diagonal()[n],np.std(conf_mat_norm[n,:])))
f.close()
# =============================================================================
# Diagonosis
# =============================================================================
h = 0 
for n in range (len(conf_mat_norm.diagonal())):
    if(conf_mat_norm.diagonal()[n]<0.3): #to check which class is lower than 30%
        #probability, directory name, number of images
        print(conf_mat_norm.diagonal()[n],list_fams[n],no_imgs[n])
        h+=1
print()        
print('%d number of classes has acc lower than 30 percent.' %h)
# =============================================================================
# Time
# =============================================================================
print()
if(time_elapsed<3600):
    print('Computation time: %f s' %time_elapsed)
else:
    print('Computation time: %f hr' %(time_elapsed/3600))





