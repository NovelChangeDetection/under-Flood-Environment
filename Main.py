import os
import cv2
import rasterio
from numpy import matlib
from AZOA import AZOA
from FOA import FOA
from GBOA import GBOA
from Glob_Vars import Glob_Vars
from Image_Results import Image_Results
from Model_GA_CNN import Model_GACNN
from Model_SCA_MCA_E_ADDUNet import Model_SCA_MCA_E_ADDUNet
from Model_TRSANet import Model_TRSANet
from Model_Unet import Model_UNet
from Objective_Function import Objfun_Cls
from PROPOSED import PROPOSED
from Plot_Resuls import *
from WGOA import WGOA


# open with rasterio
def Read_Data(path):
    with rasterio.open(path) as src:
        img = src.read(1)
    img = cv2.resize(img, [512, 512])
    return img


# Read Dataset
an = 0
if an == 1:
    path_pre = './Datasets/PRE_S1-20230517T191707Z-001/PRE_S1'
    path_post = './Datasets/POST_S1-20230517T191716Z-001/POST_S1'
    path_lab = './Datasets/Labels-20230517T191741Z-001/Labels'
    Listpre = os.listdir(path_pre)
    Listpost = os.listdir(path_post)
    Listlab = os.listdir(path_lab)
    Pre_img = []
    Post_img = []
    Label_img = []
    for k in range(len(Listpre)):
        print(k)
        image_path_pre = path_pre + '/' + Listpre[k]
        indexes1 = [i for i, v in enumerate(Listpost) if v[:-11] == Listpre[k][:-15]]
        indexes2 = [i for i, v in enumerate(Listlab) if v[:-14] == Listpre[k][:-15]]
        image_path_post = path_post + '/' + Listpost[indexes1[0]]
        image_path_lab = path_lab + '/' + Listlab[indexes2[0]]
        pre = Read_Data(image_path_pre)
        post = Read_Data(image_path_post).astype(np.uint8)
        lab = Read_Data(image_path_lab).astype(np.uint8)
        cv2.imshow('preimg', pre)
        cv2.imshow('postimg', post)
        cv2.imshow('labimg', lab)
        cv2.waitKey(0)
        Pre_img.append(pre)
        Post_img.append(post)
        Label_img.append(lab)
    np.save('Pre_Images.npy', Pre_img)
    np.save('Post_Images.npy', Post_img)
    np.save('Label_Images.npy', Label_img)

# Optimization for Change Detection
an = 0
if an == 1:
    Images1 = np.load('Pre_Images.npy', allow_pickle=True)
    Images2 = np.load('Post_Images.npy', allow_pickle=True)
    Label_img = np.load('Label_Images.npy', allow_pickle=True)
    Glob_Vars.Images1 = Images1
    Glob_Vars.Images2 = Images2
    Glob_Vars.Label = Label_img
    Npop = 10
    Chlen = 3  # Hidden Neuron Count, Learning Rate, No of Epochs
    xmin = matlib.repmat(np.asarray([5, 0.01, 5]), Npop, 1)
    xmax = matlib.repmat(np.asarray([255, 0.99, 50]), Npop, 1)
    fname = Objfun_Cls
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("GBOA...")
    [bestfit1, fitness1, bestsol1, time1] = GBOA(initsol, fname, xmin, xmax, Max_iter)  # GBOA

    print("FOA...")
    [bestfit2, fitness2, bestsol2, time2] = FOA(initsol, fname, xmin, xmax, Max_iter)  # FOA

    print("AZOA...")
    [bestfit3, fitness3, bestsol3, time3] = AZOA(initsol, fname, xmin, xmax, Max_iter)  # AZOA

    print("WGOA...")
    [bestfit4, fitness4, bestsol4, time4] = WGOA(initsol, fname, xmin, xmax, Max_iter)  # WGOA

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # PROPOSED

    Fitness = ([fitness1, fitness2, fitness3, fitness4, fitness5])
    Best_sol = ([bestsol1, bestsol2, bestsol3, bestsol4, bestsol5])

    np.save('Fitness.npy', np.asarray(Fitness)) # Save the Fitness
    np.save('BestSol.npy', np.asarray(Best_sol))  # Save the Bestsol

# Change Detection
an = 0
if an == 1:
    Images1 = np.load('Pre_Images.npy', allow_pickle=True)
    Images2 = np.load('Post_Images.npy', allow_pickle=True)
    Label_img = np.load('Label_Images.npy', allow_pickle=True)
    BestSol = np.load('BestSol.npy', allow_pickle=True)[:, 4]
    EVAL = []
    Steps_per_epoch = [50, 100, 150, 200, 250]
    for epoch in range(len(Steps_per_epoch)):
        Eval = np.zeros((10, 25))
        for j in range(BestSol.shape[0]):
            print(epoch, j)
            sol = np.round(BestSol[j, :]).astype(np.int16)
            Eval[j, :] = Model_SCA_MCA_E_ADDUNet(Images1, Images2, Label_img, Steps_per_epoch[epoch], sol=sol)
        Eval[5, :] = Model_UNet(Images1, Images2, Label_img, Steps_per_epoch[epoch])
        Eval[6, :] = Model_TRSANet(Images1, Images2, Label_img, Steps_per_epoch[epoch])
        Eval[7, :] = Model_GACNN(Images1, Images2, Label_img, Steps_per_epoch[epoch])
        Eval[8, :] = Model_SCA_MCA_E_ADDUNet(Images1, Images2, Label_img, Steps_per_epoch[epoch])
        Eval[9, :] = Eval[4, :]
        EVAL.append(Eval)
    np.save('Eval_All_Steps.npy', np.asarray(EVAL))  # Save the Eval_all

Image_Results()
plot_results_conv()
Plot_obj()
Plot_Seg_Results()
