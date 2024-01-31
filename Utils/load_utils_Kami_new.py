from Utils.gen_functions import calc_snr
import tensorflow as tf
import numpy as np
from keras.layers import Input, Conv2D, Lambda
from keras.models import Model, Sequential
from keras import backend as K
import h5py    
from Utils.vim1_data_handler import vim1_data_handler as vim1_data_handler

def get_data_from_fmrih5(h5file):
    fmri_index    = np.where(~np.isnan(h5file['metadata']['value'][0]))[0]
    session_index = np.where(~np.isnan(h5file['metadata']['value'][1]))[0][0]
    run_index     = np.where(~np.isnan(h5file['metadata']['value'][2]))[0][0]
    block_index   = np.where(~np.isnan(h5file['metadata']['value'][3]))[0][0]
    label_index   = np.where(~np.isnan(h5file['metadata']['value'][4]))[0][1]
    ### Location 19-21
   
    data    = h5file['dataset'].value
    fmri    = data[:,fmri_index]
    session = data[:,session_index]
    run     = data[:,run_index]
    block   = data[:,block_index]
    label   = data[:,label_index]
    return fmri, session, run, block, label
def get_avg_FMRI(fmri,labels):
    labels_unique = np.unique(labels)
    fmri_avg = np.zeros((len(labels_unique),fmri.shape[1]))
    for l in range(len(labels_unique)):
        fmri_cur = fmri[labels ==labels_unique[l]]
        fmri_avg[l,:] = np.expand_dims(fmri_cur.mean(0),0)
    return fmri_avg,labels_unique.astype(int)
def normalize_FMRI(fmri_used,fmri_output):
    fmri = (fmri_output - fmri_used.mean(0))/fmri_used.std(0)
    # fmri = fmri / fmri.std(0)
    return fmri
def normalize_FMRI_per_session(fmri_used,fmri_output,session):
    fmri = fmri_output.copy()
    session_unique = np.unique(session)
    for s in session_unique:
        ind = (session == s)
        fmri[ind,:] = (fmri_output[ind,:] - fmri_used[ind,:].mean(0))/fmri_used[ind,:].std(0)
    return fmri
def normalize_FMRI_ver2(fmri):
    fmri = fmri - fmri.mean(0)
    fmri = fmri / np.abs(fmri).max(0)
    return fmri

def get_ROI_from_fmrih5(f1):
    fmri_index    = np.where(~np.isnan(f1['metadata']['value'][0]))[0]
    num_voxels = fmri_index[-1]
    ROI_V1  = f1['metadata']['value'][22][:num_voxels+1]
    ROI_V2  = f1['metadata']['value'][23][:num_voxels+1]
    ROI_V3  = f1['metadata']['value'][24][:num_voxels+1]
    ROI_hV4 = f1['metadata']['value'][25][:num_voxels+1]
    ROI_LOC = f1['metadata']['value'][26][:num_voxels+1]
    ROI_FFA = f1['metadata']['value'][27][:num_voxels+1]
    ROI_PPA = f1['metadata']['value'][28][:num_voxels+1]
    ROI_LVC = (ROI_V1 + ROI_V2 + ROI_V3 + ROI_hV4) == 1
    ROI_HVC = (ROI_LOC + ROI_FFA + ROI_PPA) == 1
    ROI_VC = (ROI_LVC + ROI_HVC) == 1
    return ROI_V1, ROI_V2, ROI_V3, ROI_hV4, ROI_LOC, ROI_FFA, ROI_PPA,ROI_LVC,ROI_HVC,ROI_VC

def take_part_of_data(y,y_labels,num_sample,take_part_out = 0):
        # If we only need a sample size smaller than 1200, we choose from the first repetition.
    if num_sample < 1200:
        rep = 1
    else:
        rep = int(num_sample/1200)

    # select the needed repetitions.
    tmp = np.zeros(5,dtype=bool)
    tmp[:rep] = True
    sel = np.tile(tmp, 1200)
    if(take_part_out != 0):
        sel[-take_part_out * 5:] = False
    y_labels = y_labels[sel]
    y = y[sel]

    # If we only need a sample size smaller than 1200, samples belongs to different categories are chosen to avoid any bias.
    # Here we have 150 image categories, 8 images per category 
    if num_sample==300:
        # 2 images per category
        y = y[0::4]
        y_labels = y_labels[0::4]    
    
    elif num_sample==600:
        # 4 images per category
        y = np.vstack((y[0::4], y[1::4]))
        y_labels = np.concatenate((y_labels[0::4], y_labels[1::4]),0)

    elif num_sample==900:   
        # 6 images per category
        y = np.vstack((y[0::4], y[1::4], y[2::4]))
        y_labels = np.concatenate((y_labels[0::4], y_labels[1::4], y_labels[2::4]),0)
    return y,y_labels
def get_data_newK(subject,include_roi = 0,get_train_avg= 0,num_sample = 6000,Voxel_to_choose = 0, include_ind_tresh_snr = 0,take_part_out =0):
    f1 = h5py.File('data/Processed_data/GOD/sub-0' + str(subject) +'_NaturalImageTraining.h5','r')    
    fmri_train, session_train, run, block, label_train = get_data_from_fmrih5(f1)
    f2 = h5py.File('data/Processed_data/GOD/sub-0' + str(subject) +'_NaturalImageTest.h5','r')    
    fmri_test, session_test, run, block, label_test = get_data_from_fmrih5(f2)
    # fmri_train = normalize_FMRI_per_session(fmri_train,fmri_train,session_train)
    # fmri_test = normalize_FMRI_per_session(fmri_test,fmri_test,session_test)
    fmri_train = normalize_FMRI(fmri_train,fmri_train)
    fmri_test = normalize_FMRI(fmri_test,fmri_test)
    if(Voxel_to_choose == 0):
        Voxel_to_choose = fmri_train.shape[1]
    Y = fmri_train; Y_test = fmri_test; 
    
    Y_test_avg,test_avg_labels = get_avg_FMRI(fmri_test,label_test)
    labels_train = label_train.astype(int) - 1; labels = label_test.astype(int)- 1
    Y_test_median = Y_test_avg

    labels_index = np.argsort(labels_train.flatten())
    labels_train = labels_train[labels_index]
    Y = Y[labels_index,:]

    # Get image data
    dir = 'data/Processed_data/GOD/'
    file= np.load(dir+'ext_images_test_112.npz')
    ext_img_test = file['img_112']

    file= np.load(dir+'images_112.npz') #_56

    X = file['train_images']
    X_test = file['test_images']
    X_test_sorted = X_test

    X_test = X_test[labels]
    snr  = calc_snr(Y_test,Y_test_avg,labels)
    snr_inv = 1/snr


    snr = snr/snr.mean()
    ind_tresh_snr = np.argsort(snr)[-Voxel_to_choose:]
    snr = snr[ind_tresh_snr]
    snr_inv = snr_inv[ind_tresh_snr]
    snr_inv = snr_inv/snr_inv.mean()

    voxel_loc = 0
    SNR  = tf.constant(snr,shape = [1,len(snr)],dtype = tf.float32)
    #repetition choosing

    Y,labels_train = take_part_of_data(Y,labels_train,num_sample,take_part_out = take_part_out)
    if(get_train_avg):
        Y,labels_train = get_avg_FMRI(Y,labels_train)
    X = file['train_images']
    X = X[labels_train]

    ROI_V1, ROI_V2, ROI_V3, ROI_hV4, ROI_LOC, ROI_FFA, ROI_PPA,ROI_LVC,ROI_HVC,ROI_VC = get_ROI_from_fmrih5(f1)
    if(include_roi):
        return X,X_test,X_test_sorted,Y[:,ind_tresh_snr], Y_test[:,ind_tresh_snr], Y_test_avg[:,ind_tresh_snr],Y_test_median[:,ind_tresh_snr],\
            labels_train, labels,ext_img_test,SNR,snr_inv,snr,voxel_loc,\
            ROI_V1[ind_tresh_snr], ROI_V2[ind_tresh_snr], ROI_V3[ind_tresh_snr], ROI_hV4[ind_tresh_snr], ROI_LOC[ind_tresh_snr]\
                , ROI_FFA[ind_tresh_snr], ROI_PPA[ind_tresh_snr],ROI_LVC[ind_tresh_snr],ROI_HVC[ind_tresh_snr],ROI_VC[ind_tresh_snr]
    else:
        if (include_ind_tresh_snr):
            return X,X_test,X_test_sorted,Y[:,ind_tresh_snr], Y_test[:,ind_tresh_snr], Y_test_avg[:,ind_tresh_snr],\
                    Y_test_median[:,ind_tresh_snr],labels_train, labels,ext_img_test,SNR,snr_inv,snr,voxel_loc,ind_tresh_snr
        else:
            return X,X_test,X_test_sorted,Y[:,ind_tresh_snr], Y_test[:,ind_tresh_snr], Y_test_avg[:,ind_tresh_snr],\
                Y_test_median[:,ind_tresh_snr],labels_train, labels,ext_img_test,SNR,snr_inv,snr,voxel_loc



def get_data_vim(subject):


    norm = 1
    num_voxels = 5000
    dir ='/net/mraid11/export/data/romanb/vim1/'
    data = np.load(dir+'sub1_112.npz')
    X             =  data['X_train_3']
    X_test        =  data['X_test_3']
    X_test_sorted = X_test
    handler = vim1_data_handler(norm = norm ,select_areas =None ,select_by_snr = 1,num_voxels = num_voxels,log=True)
    data_dict = handler.get_data()
    Y_test_avg = data_dict['test_avg']
    Y = data_dict['train_avg']
    labels = data_dict['test_label']
    Y_test = data_dict['test']

    snr  = calc_snr(Y_test,Y_test_avg,labels)
    snr_inv = 1/snr
    voxel_loc = 0
    SNR  = tf.constant(snr,shape = [1,len(snr)],dtype = tf.float32)
    #repetition choosing

    return X,X_test,X_test_sorted,Y, Y_test, Y_test_avg, labels, SNR,snr_inv,snr,voxel_loc