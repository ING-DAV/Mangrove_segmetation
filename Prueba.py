import cv2 as cv  #Open Cv library, has utilities for image processing
import numpy as np #Native python library for Math operations
from skimage.feature import graycomatrix as glcm #Import gray scale co-occurrence matrix, computes gray levels around a set of pixel
from skimage.feature import graycoprops as glcm_p #Computes properties of the gray scale co-occurrence matrix
from skimage.feature import local_binary_pattern as lbp #Local binary pattern. Computes binary arrays based on the pixel distribution
from sklearn import svm #Support Vector Machine. It calculates a vector that separates classes within a defined space
from sklearn.svm import LinearSVC #Machine Learning utilities library
import multiprocessing #Multi CPU processing
import pandas as pd  #Data manipulation utilities library
import glob #Library for listing directories and files
import time # Library to keep track on the computing elapsed time
from multiprocessing import Pool # Generates a procdssing thread
from concurrent.futures import ProcessPoolExecutor as PPE # To distribute processing tasks
from sklearn.model_selection import GridSearchCV #Lazy Evaluation to select the model
import seaborn as sns #Plot generation library
from matplotlib import pyplot as plt #Plot generation library



def features(im_s):
    
    n_params = 7

    params = np.zeros(n_params)

    mtxD = glcm(im_s,distances = [1],angles = [0], levels = 256, symmetric = True, normed = True)
    mtxI = glcm(im_s,distances = [1],angles = [np.pi], levels = 256, symmetric = True, normed = True)
    mtxDd = glcm(im_s,distances = [1],angles = [np.pi/2], levels = 256, symmetric = True, normed = True)
    mtxDi = glcm(im_s,distances = [1],angles = [-np.pi/2], levels = 256, symmetric = True, normed = True)
    mtxDi1 = glcm(im_s,distances = [1],angles = [np.pi/4], levels = 256, symmetric = True, normed = True)
    mtxDi2 = glcm(im_s,distances = [1],angles = [-np.pi/4], levels = 256, symmetric = True, normed = True)
    
    params[0] = glcm_p(mtxD,prop = 'contrast') ##
    #params[1] = glcm_p(mtxI,prop = 'dissimilarity')
    #params[2] = glcm_p(mtxDd,prop = 'homogeneity')
    #params[3] = glcm_p(mtxDi,prop = 'ASM')
    #params[4] = glcm_p(mtxD,prop = 'energy')
    #params[5] = glcm_p(mtxD,prop = 'correlation')

    params[1] = glcm_p(mtxI,prop = 'contrast')##
    #params[7] = glcm_p(mtxI,prop = 'dissimilarity')
    #params[8] = glcm_p(mtxI,prop = 'homogeneity')
    #params[9] = glcm_p(mtxI,prop = 'ASM')
    #params[10] = glcm_p(mtxI,prop = 'energy')
    #params[11] = glcm_p(mtxI,prop = 'correlation')

    params[2] = glcm_p(mtxDd,prop = 'contrast')##
    #params[13] = glcm_p(mtxDd,prop = 'dissimilarity')
    #params[14] = glcm_p(mtxDd,prop = 'homogeneity')
    #params[15] = glcm_p(mtxDd,prop = 'ASM')
    #params[16] = glcm_p(mtxDd,prop = 'energy')
    #params[17] = glcm_p(mtxDd,prop = 'correlation')
    
    params[3] = glcm_p(mtxDi,prop = 'contrast') ##
    #params[19] = glcm_p(mtxDi,prop = 'dissimilarity')
    #params[20] = glcm_p(mtxDi,prop = 'homogeneity')
    #params[21] = glcm_p(mtxDi,prop = 'ASM')
    #params[22] = glcm_p(mtxDi,prop = 'energy')
    #params[23] = glcm_p(mtxDi,prop = 'correlation')

    params[4] = glcm_p(mtxDi1,prop = 'contrast')##
    #params[25] = glcm_p(mtxDi1,prop = 'dissimilarity')
    #params[26] = glcm_p(mtxDi1,prop = 'homogeneity')
    #params[27] = glcm_p(mtxDi1,prop = 'ASM')
    #params[28] = glcm_p(mtxDi1,prop = 'energy')
    #params[29] = glcm_p(mtxDi1,prop = 'correlation')   

    params[5] = glcm_p(mtxDi2,prop = 'contrast')##     
    #params[31] = glcm_p(mtxDi2,prop = 'dissimilarity')     
    #params[32] = glcm_p(mtxDi2,prop = 'homogeneity')     
    #params[33] = glcm_p(mtxDi2,prop = 'ASM')     
    #params[34] = glcm_p(mtxDi2,prop = 'energy') 
    #params[35] = glcm_p(mtxDi2,prop = 'correlation')
# #
    params[6] = np.mean(im_s) ##
    


    #params[25] = 
        #print('Params: ',params)
    return params

def lbp_hg(img):

   global n_bins

   eps = 1e-7
   radius = 10
   n_points = 64 * radius
   n_bins = 128

   im_g = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
   lb = lbp(im_g,n_points,radius, method = 'uniform')
   (hg, _) = np.histogram(lb.ravel(),bins = 128, range =(0,256) )
   hg = hg.astype("float")
   hg /= (hg.sum() + eps)

   return hg




def n_wins(imr, W):
    a = imr.shape[0]
    h = imr.shape[1]

    n1 = a // W
    n2 = h // W



    nW = n1 * n2

    if nW > 16:
      n1 = 4
      n2 = 4
      nW = 16 

    return nW, n1, n2





def train_w(im_w,W):

        n_feats = 7*3

        feats = np.zeros(n_feats)

        for i in range(0,3):
                  
                  feats[i*(n_feats//3):(i*(n_feats//3) + (n_feats//3))] = features(im_w[:,:,i])#j*W:(j*W)+W, j*W:(j*W)+W, i])
        
        return feats

def train_t(im_tr):

    n_feats = 7*3

    Wx = im_tr.shape[0]//10
    Wy = im_tr.shape[1]//10
    
    tr = np.zeros((10,n_feats))


    
    for i in range(0, 10):
      for j in range(0, 10):
        for k in range(0,3):
           tr[i,k*(n_feats//3):(k*(n_feats//3))+(n_feats//3)] = features(im_tr[i*Wx:(i*Wx)+Wx, j*Wy:(j*Wy)+Wy,k ])
           #tr[i:,k*(n_feats//3):(k*(n_feats//3))+(n_feats//3)] = features(im_tr[:,:,k])
    
    #print(tr)

    return tr


def train():
   global lin_svm
   #global poly_svm

   
   train_rojo = glob.glob("Training/RA/RA*")
   t_rojo = len(train_rojo)
   print('Total Rojo: ',t_rojo)

   train_rojo_bb = glob.glob("Training/RBB/RBB*")
   t_rojo_bb = len(train_rojo_bb)
   print('Total Rojo BB: ',t_rojo_bb)

   train_bot = glob.glob("Training/BOT/BOT*")
   t_bot = len(train_bot)
   print('Total Bot: ',t_bot)

   train_blanco = glob.glob("Training/BL/Bl*")
   t_bl = len(train_blanco)
   print('Total BL: ',t_bl)

   train_nomangle = glob.glob("Training/NM/NM*")
   t_nm = len(train_nomangle)
   print('Total NO_MANGLE: ',t_nm)

   

   train_fil = train_rojo + train_rojo_bb + train_bot + train_blanco + train_nomangle
   total_train = len(train_fil)
   print('Total files: ', total_train)

   global ls_t

   ls_t = [t_rojo, t_rojo_bb, t_bot, t_bl,t_nm]

   
   #ax_t = []



   
   n_feats = 7*3
   x = np.zeros((total_train*10, n_feats))#+n_bins))

   ctr = 0
   ctr1 = 0
   ctr2 = 0 


   

   for i in train_fil:

     im_tr = cv.imread(i)
     print('Train: ',ctr,i)
     x[ctr:ctr+10,:] = train_t(im_tr)
     #x[ctr:ctr+10,n_feats:n_feats+n_bins] = lbp_hg(im_tr)
     
     ctr = ctr + 10

   df = pd.DataFrame(x)
   df.to_csv('Vectores.csv')
   
   
   print('Training vectors complete')

   #print(tots)
   print(ctr)

   l1 = [1]*10*t_rojo#int(tots[0])
   l2 = [2]*10*t_rojo_bb#int((tots[1]-tots[0]))
   l3 = [3]*10*t_bot#int((tots[2]-tots[1]))
   l4 = [4]*10*t_bl#int((tots[3]+tots[2]))
   l5 = [5]*10*t_nm#int((ctr-tots[3]-tots[2]-tots[1]-tots[0]))

   y = l1 + l2 + l3 + l4 + l5 

   if len(x) > len(y):
      x = x[0:len(y)]
   elif len(y) > len(x):
      y = y[0:len(x)]


   lin_svm = LinearSVC(dual= False, C = 0.1)
   lin_svm.fit(x,y)

   #poly_svm = LinearSVC(dual=False, C = 1)
   #poly_svm.fit(x,y)


   return lin_svm


def test(im_test):

   global im_ax1
   #global im_ax2


   im_ax1 = np.zeros((im_test.shape[0],im_test.shape[1]),np.uint8)
   #im_ax2 = np.zeros((im_test.shape[0],im_test.shape[1]),np.uint8)

   n_feats = 7*3
   caracs = np.zeros(n_feats)#+n_bins)

  

   for i in range(0,im_test.shape[0],W):
    for j in range(0,im_test.shape[1],W):

        ROI = im_test[i:i+(W),j:j+(W),:]
        caracs[0:n_feats] = train_w(ROI,W)
        #caracs[n_feats:n_feats+n_bins]=lbp_hg(ROI)


        qw1 = lin_svm.predict(caracs.reshape(-1,1).T)
        #qw2 = poly_svm.predict(caracs.reshape(-1,1).T)

        im_ax1[i:i+(W),j:j+(W)] = qw1*50
        #im_ax2[i:i+W,j:j+W] = qw2*50

   return im_ax1

def masks(im):
    
    mask1 = np.zeros((im.shape[0],im.shape[1]),np.uint8)
    mask2 = np.zeros((im.shape[0],im.shape[1]),np.uint8)
    mask3 = np.zeros((im.shape[0],im.shape[1]),np.uint8)
    mask4 = np.zeros((im.shape[0],im.shape[1]),np.uint8)
    mask5 = np.zeros((im.shape[0],im.shape[1]),np.uint8)

    #mask6 = np.zeros((im.shape[0],im.shape[1]),np.uint8)
    #mask7 = np.zeros((im.shape[0],im.shape[1]),np.uint8)
    #mask8 = np.zeros((im.shape[0],im.shape[1]),np.uint8)
    #mask9 = np.zeros((im.shape[0],im.shape[1]),np.uint8)
    #mask10 = np.zeros((im.shape[0],im.shape[1]),np.uint8)
    
    for i in range(0,im.shape[0],):
        for j in range(0,im.shape[1]):

          #print(ax[i,j])  
          
          if ax1[i,j] > 40 and ax1[i,j] < 60:
            mask1[i,j] = 1

          elif ax1[i,j] > 90 and ax1[i,j] < 110:
            mask2[i,j] = 1

          elif ax1[i,j] > 140 and ax1[i,j] < 160 :  
            mask3[i,j] = 1 

          elif ax1[i,j] > 190 and ax1[i,j] < 210 :  
            mask4[i,j] = 1 

          elif ax1[i,j] > 240:
            mask5[i,j] = 1

    #for i in range(0,im.shape[0]):
    #  for j in range(0,im.shape[1]):

        #if ax2[i,j] > 40 and ax2[i,j] < 60:
        #  mask6[i,j] = 1

        #elif ax2[i,j] > 90 and ax2[i,j] < 110:
        #  mask7[i,j] = 1

        #elif ax2[i,j] > 140 and ax2[i,j] < 160 :  
        #  mask8[i,j] = 1 

        #elif ax2[i,j] > 190 and ax2[i,j] < 210 :  
        #  mask9[i,j] = 1 

        #elif ax2[i,j] > 240:
        #   mask10[i,j] = 1



    im_f1 = cv.bitwise_and(im,im,mask = mask1)
    im_f2 = cv.bitwise_and(im,im,mask = mask2)
    im_f3 = cv.bitwise_and(im,im,mask = mask3)
    im_f4 = cv.bitwise_and(im,im,mask = mask4)
    im_f5 = cv.bitwise_and(im,im,mask = mask5)

    #im_f6 = cv.bitwise_and(im,im,mask = mask6)
    #im_f7 = cv.bitwise_and(im,im,mask = mask7)
    #im_f8 = cv.bitwise_and(im,im,mask = mask8)
    #im_f9 = cv.bitwise_and(im,im,mask = mask9)
    #im_f10 = cv.bitwise_and(im,im,mask = mask10)

    return im_f1, im_f2, im_f3, im_f4, im_f5 #im_f6, im_f7, im_f8, im_f9, im_f10


def no_veg(im_o,W):
   train_nv = glob.glob("Training/NV/NV*")
   t_nv = len(train_nv)
   print('Total NV: ',t_nv)

   train_veg = glob.glob("Training/Veg/*")
   t_veg = len(train_veg)
   print('Total Vegetación: ',t_veg)
   
   t_f = train_nv + train_veg

   tot = len(t_f)

   n_feats = 7*3
   x_nv = np.zeros((tot*10, n_feats))#+n_bins))  

   

   ctr = 0
   for i in t_f:

     im_tr = cv.imread(i)
     print('Train: ',ctr,i)
     x_nv[ctr:ctr+10,:] = train_t(im_tr)
     #x[ctr:ctr+10,n_feats:n_feats+n_bins] = lbp_hg(im_tr)
     
     ctr = ctr + 10

   df_nv = pd.DataFrame(x_nv)
   df_nv.to_csv('Vectores_NV.csv')


   l1 = [1]*10*t_nv
   l2 = [2]*10*t_veg

   y_nv = l1 + l2

   print('Veg vectors complete.')
   
   nv_svm = svm.SVC(kernel = 'linear',C=0.1, gamma = 100)
   nv_svm.fit(x_nv,y_nv)

   nv_ax1 = np.zeros((im_o.shape[0],im_o.shape[1]),np.uint8)

   n_feats = 7*3
   #n_bins = 128s
   caracs_nv = np.zeros(n_feats)#+n_bins)

   print('Train veg complete.')



  

   for i in range(0,im_o.shape[0],W):
    for j in range(0,im_o.shape[1],W):

        ROI = im_o[i:i+(W),j:j+(W),:]
        caracs_nv = train_w(ROI,W)
        #caracs[n_feats:n_feats+n_bins]=lbp_hg(ROI)


        qw_nv = nv_svm.predict(caracs_nv.reshape(-1,1).T)
        

        nv_ax1[i:i+(W),j:j+(W)] = qw_nv*50


   print('Veg test complete.')

   mask_nv = np.zeros((im_o.shape[0],im_o.shape[1]),np.uint8)
   mask_v = np.zeros((im_o.shape[0],im_o.shape[1]),np.uint8)

   for i in range(0,im_o.shape[0],):
      for j in range(0,im_o.shape[1]):
        if nv_ax1[i,j] > 40 and nv_ax1[i,j] < 60:
            mask_nv[i,j] = 1
        else: 
            mask_v [i,j] = 1

   nv_f1 = cv.bitwise_and(im_o,im_o,mask = mask_nv)
   nv_f2 = cv.bitwise_and(im_o,im_o,mask = mask_v)

   print('Veg masks complete.')

   nv_f1_s = cv.resize(nv_f1,(500,500),interpolation = cv.INTER_AREA)
   nv_f2_s = cv.resize(nv_f2,(500,500),interpolation = cv.INTER_AREA)

   cv.imwrite('Segmentation/No_vegetacion.jpg',nv_f1_s)
   cv.imwrite('Segmentation/Vegetacion.jpg',nv_f2_s)

   #cv.imshow('NV',cv.resize(nv_f1,(1000,1000),interpolation=cv.INTER_AREA))
   #cv.imshow('V',cv.resize(nv_f2,(1000,1000),interpolation=cv.INTER_AREA))
   
   #cv.waitKey(0)



   return nv_f1, nv_f2
         
def main():


   global W 
   global ax1
   global ax2 

   num_cores = multiprocessing.cpu_count()
   print("Available cores: ", num_cores)

   st = time.time()

   W =10

   im = cv.imread('Maha/Puerto.tif')

   #im = im[1000:8000,1000:11000]
   im1 = cv.resize(im,(500,500),interpolation = cv.INTER_AREA)
   
   im_nv, im_v = no_veg(im,W) 
   et_veg = time.time()
   print("Elapsed time for veg/no veg: ", (et_veg - st)/60, " minutes." )

   sv1 = train()
   print('Train complete')
   et_train = time.time() 
   print("Elapsed time for training: ", (et_train - st)/60, " minutes." )

   ax1 = test(im_v)
   
   Sim_ax1 = cv.resize(ax1,(500,500),interpolation = cv.INTER_AREA)
   #Sim_ax2 = cv.resize(ax2,(500,500),interpolation = cv.INTER_AREA)
   
   et_test = time.time()
   print("Elapsed time for testing: ", (et_test - st)/60, " minutes."  )
   print('Test complete')

   ax01, ax02, ax03, ax04, ax05 = masks(im_v) 

   cv.imshow('Lin',Sim_ax1)
   #cv.imshow('Poly',Sim_ax2)

   cv.imwrite('Segmentation/Lineal.jpg',Sim_ax1)
   #cv.imwrite('Segmentation/Poly.jpg',Sim_ax2)

   cv.imshow('ROJO_ALTO_L',cv.resize(ax01,(1000,1000),interpolation=cv.INTER_AREA))
   cv.imshow('ROJO_BB_L',cv.resize(ax02,(1000,1000),interpolation=cv.INTER_AREA))
   cv.imshow('BOTONCILLO_L',cv.resize(ax03,(1000,1000),interpolation=cv.INTER_AREA))
   cv.imshow('BLANCO_L',cv.resize(ax04,(1000,1000),interpolation=cv.INTER_AREA))
   cv.imshow('NO_MANGLE_L',cv.resize(ax05,(1000,1000),interpolation=cv.INTER_AREA))

   #cv.imshow('ROJO_ALTO_P',cv.resize(ax06,(1000,1000),interpolation=cv.INTER_AREA))
   #cv.imshow('ROJO_BB_P',cv.resize(ax07,(1000,1000),interpolation=cv.INTER_AREA))
   #cv.imshow('BOTONCILLO_P',cv.resize(ax08,(1000,1000),interpolation=cv.INTER_AREA))
   #cv.imshow('BLANCO_P',cv.resize(ax09,(1000,1000),interpolation=cv.INTER_AREA))
   #cv.imshow('NO_MANGLE_P',cv.resize(ax10,(1000,1000),interpolation=cv.INTER_AREA))

   cv.imwrite('Segmentation/rojo_alto_L.tif',ax01)
   cv.imwrite('Segmentation/rojo_bb_L.tif',ax02)
   cv.imwrite('Segmentation/botoncillo_L.tif',ax03)
   cv.imwrite('Segmentation/blanco_L.tif',ax04)
   cv.imwrite('Segmentation/no_mangle_L.tif',ax05)

   #cv.imwrite('Segmentation/rojo_alto_P.tif',ax06)
   #cv.imwrite('Segmentation/rojo_bb_P.tif',ax07)
   #cv.imwrite('Segmentation/botoncillo_P.tif',ax08)
   #cv.imwrite('Segmentation/blanco_P.tif',ax09)
   #cv.imwrite('Segmentation/no_mangle_P.tif',ax10)

   et_end = time.time()
   print("Elapsed total time: ", (et_end - st)/60, " minutes." )
   
   
   cv.imshow('Original',im1)
   #cv.imwrite('Segmentation/Original.jpg',im1)
   cv.waitKey(0)
   

if __name__ == '__main__':
    main()
 
##TQM
##TQM##TQM##TQM##TQM##TQM##TQM##TQM