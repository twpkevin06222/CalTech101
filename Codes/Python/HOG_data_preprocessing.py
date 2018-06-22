# =============================================================================
# LIbraries
# =============================================================================
import cv2
import glob
import os

# =============================================================================
# Gray Scale 
# =============================================================================
main_dir = os.listdir("C:/Users/blaze03/Desktop/CalTech101") #file directory 
def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray

for dir in main_dir:
    class_dir = ("C:/Users/blaze03/Desktop/CalTech101/%s/*.jpg" %dir)
    cv_img = [] 
    scr_file = glob.glob(class_dir)
    i = len(scr_file)
    for img in scr_file:
        n = cv2.imread(img)
        cv_img.append(n)                #color image
        
    ct101_gray = [] 
    
    os.makedirs('C:/Users/blaze03/Desktop/New/%s'%dir)
    path_g = 'C:/Users/blaze03/Desktop/New/%s'%dir
    for x in range(i):
        g = to_gray(cv_img[x])
        g_re = cv2.resize(g,(256,256))       #resizing need to be consider!
        ct101_gray.append(g_re)            #gray scale image
        #change path_g to the variable you declare your path
        cv2.imwrite(os.path.join(path_g , "img_gray_%d.jpg" %x), ct101_gray[x])
    cv2.waitKey(0)
    
