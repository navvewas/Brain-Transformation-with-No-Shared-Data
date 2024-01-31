import h5py    
import numpy as np 
import csv
import torch
import os.path
import scipy.io
import nibabel as nib
from scipy.misc import imresize
from PIL import Image
# import torch.nn as nn
from scipy.ndimage import convolve
from Utils.params.param_NSD_load import *
import json

def Load_NSD_data(cur_subj_num,save_dir = save_dir,func_type = 'fsaverage', res = 0):
        ### Load all the needed NSD data
        #  Input -  subject number(cur_subj_num)
        # Outputs:
        # 'FMRI_data' - FMRI Voxel data(Expirments x voxels); FMRI_voxels_cordinates - x,y,z cordinates of each voxel
        # 'Images_cropped_data' - Cropped images by the order of the FMRI; Image_labels (73k); 'NCSNR' Noise ceiling SNR
        # 'FMRI_ordered_10k' - FMRI voxel data ordered according to 
        ###
        # FMRI_ordered_10k, Images_ordered_10k,Images_ordered_10k_labels,FMRI_Averaged
        # Load the data
        if(res != 0):
                res_text = '_res_' + str(res)
                divide_num = 255
        else:
                res_text = ''
                divide_num = 1

        # FMRI_data = torch.load(save_dir+'FMRI_data_subj0'+str(cur_subj_num)+ '_' + func_type + '_' + processing_type+'.pt')
        # ind_out = torch.nonzero(FMRI_data.std(0) != 0) # exclude constants voxels
        # FMRI_data = FMRI_data[:,ind_out[:,0]]
        # max_ind = FMRI_data.shape[0]
        # Images_cropped_data = (torch.load(save_dir+'Images_ordered_data_subj0' +str(cur_subj_num)+ res_text +'.pt')/divide_num)[:max_ind]
        # Image_labels = torch.load(save_dir+'Images_ordered_label_subj0'+str(cur_subj_num)+'.pt')[0,:max_ind]
        # NCSNR = torch.load(save_dir+'NCSNR_subj0'+str(cur_subj_num)+ '_' + func_type + '_' + processing_type+'.pt')
        # NCSNR = NCSNR[ind_out[:,0]]
    
        FMRI_ordered_10k = torch.load(save_dir+'FMRI_ordered_10k_subj0'+str(cur_subj_num)+ '_' + func_type + '_' + processing_type+'.pt')
        ind_out = torch.nonzero(FMRI_ordered_10k.std(0) != 0)
        FMRI_ordered_10k = FMRI_ordered_10k[:,ind_out[:,0]]
        Images_ordered_10k = torch.load(save_dir+'Images_ordered_10k_subj0'+str(cur_subj_num)+ '_' + func_type + '_' + processing_type + res_text + '.pt')/divide_num
        Images_ordered_10k_labels = torch.transpose(torch.load(save_dir+'Images_ordered_10k_labels_subj0'+str(cur_subj_num)+ '_' + func_type + '_' + processing_type+'.pt'),0,1)

        Shared_mask = torch.load(save_dir+'shared_mask_subj0'+str(cur_subj_num)+'.pt')
        Shared_mask_ordered_10k = torch.load(save_dir+'shared_mask_ordered_10k_subj0'+str(cur_subj_num)+ '_' + func_type + '_' + processing_type+'.pt')
        including_ind_mask = torch.load(save_dir+'iculding_ind_mask_ordered_10k_subj0'+str(cur_subj_num)+ '_' + func_type + '_' + processing_type+'.pt')
        
        train_mask_s = torch.load(save_dir+'train_mask_shared.pt')
        val_mask_s = torch.load(save_dir+'val_mask_shared.pt')
        test_mask_s =torch.load(save_dir+'test_mask_shared.pt')
        train_mask = torch.load(save_dir+'train_mask.pt')
        val_mask = torch.load(save_dir+'val_mask.pt')
        test_mask = torch.load(save_dir+'test_mask.pt')
        shared_mask = torch.load(save_dir+'shared_mask.pt')
        shared_all_indexes = torch.load(save_dir + 'shared_all_indexes_subj0'+str(cur_subj_num) + '.pt')

        return FMRI_ordered_10k.numpy(), Images_ordered_10k.numpy(),\
                        Images_ordered_10k_labels.numpy(),\
                        train_mask_s,val_mask_s,test_mask_s,train_mask,val_mask,test_mask,shared_mask,shared_all_indexes

def Split_data_coco_labels(coco_labels,Image_labels,train_mask,val_mask,test_mask,shared_all_indexes = [],\
        Image_labels_second_subj = [],FMRI_averaged_all = []):
        Image_labels = Image_labels.astype(int)
        if(len(Image_labels_second_subj) != 0):
                Image_labels_second_subj = Image_labels_second_subj.astype(int)
                res = np.intersect1d(Image_labels[np.nonzero(train_mask[Image_labels])[0]],\
                         Image_labels_second_subj[np.nonzero(train_mask[Image_labels_second_subj])[0]])
                res2,train_ind,ind = np.intersect1d(Image_labels,res,return_indices=True)
                res = np.intersect1d(Image_labels[np.nonzero(val_mask[Image_labels])[0]],\
                         Image_labels_second_subj[np.nonzero(val_mask[Image_labels_second_subj])[0]])
                res2,val_ind,ind = np.intersect1d(Image_labels,res,return_indices=True)
                res = np.intersect1d(Image_labels[np.nonzero(test_mask[Image_labels])[0]],\
                         Image_labels_second_subj[np.nonzero(test_mask[Image_labels_second_subj])[0]])       
                res2,test_ind,ind = np.intersect1d(Image_labels,res,return_indices=True)     
        else:
                train_ind = np.nonzero(train_mask[Image_labels])
                val_ind = np.nonzero(val_mask[Image_labels])
                test_ind = np.nonzero(test_mask[Image_labels])
        train_coco_labels = coco_labels[train_ind]
        val_coco_labels = coco_labels[val_ind]
        test_coco_labels = coco_labels[test_ind]
        if (len(shared_all_indexes) != 0 ):
                test_coco_labels_shared_across = coco_labels[shared_all_indexes]
        if (len(shared_all_indexes) != 0 ):
                return train_coco_labels,val_coco_labels,test_coco_labels,test_coco_labels_shared_across
        else:
                return train_coco_labels,val_coco_labels,test_coco_labels       


def Split_data_new(Images_cropped_data,FMRI_data,Image_labels,train_mask,val_mask,test_mask,shared_all_indexes = [],\
        Image_labels_second_subj = [],FMRI_averaged_all = [],coco_labels =[]):
        Image_labels = Image_labels.astype(int)
        if(len(Image_labels_second_subj) != 0):
                Image_labels_second_subj = Image_labels_second_subj.astype(int)
                res = np.intersect1d(Image_labels[np.nonzero(train_mask[Image_labels])[0]],\
                         Image_labels_second_subj[np.nonzero(train_mask[Image_labels_second_subj])[0]])
                res2,train_ind,ind = np.intersect1d(Image_labels,res,return_indices=True)
                res = np.intersect1d(Image_labels[np.nonzero(val_mask[Image_labels])[0]],\
                         Image_labels_second_subj[np.nonzero(val_mask[Image_labels_second_subj])[0]])
                res2,val_ind,ind = np.intersect1d(Image_labels,res,return_indices=True)
                res = np.intersect1d(Image_labels[np.nonzero(test_mask[Image_labels])[0]],\
                         Image_labels_second_subj[np.nonzero(test_mask[Image_labels_second_subj])[0]])       
                res2,test_ind,ind = np.intersect1d(Image_labels,res,return_indices=True)     
        else:
                train_ind = np.nonzero(train_mask[Image_labels])
                val_ind = np.nonzero(val_mask[Image_labels])
                test_ind = np.nonzero(test_mask[Image_labels])
        train_images = Images_cropped_data[train_ind]
        val_images = Images_cropped_data[val_ind]
        test_images = Images_cropped_data[test_ind]
        train_FMRI = FMRI_data[train_ind]
        val_FMRI = FMRI_data[val_ind]
        test_FMRI = FMRI_data[test_ind]
        train_labels = Image_labels[train_ind]
        val_labels = Image_labels[val_ind]
        test_labels = Image_labels[test_ind]
        if (len(shared_all_indexes) != 0 ):
                test_FMRI_shared_across = FMRI_data[shared_all_indexes]
                test_images_shared_across = Images_cropped_data[shared_all_indexes]
                # train_FMRI_shared_across = FMRI_data[shared_all_indexes]
                # train_images_shared_across = Images_cropped_data[shared_all_indexes]
        if (len(FMRI_averaged_all) != 0 ):
                train_FMRI_averaged_all = FMRI_averaged_all[train_ind]
                val_FMRI_averaged_all = FMRI_averaged_all[val_ind]
                test_FMRI_averaged_all = FMRI_averaged_all[test_ind]
                if (len(shared_all_indexes) != 0 ):
                        return train_images,val_images,test_images,train_FMRI,val_FMRI,\
                                test_FMRI,train_labels,val_labels, test_labels,test_FMRI_shared_across,test_images_shared_across,train_FMRI_averaged_all,val_FMRI_averaged_all,test_FMRI_averaged_all
                return train_images,val_images,test_images,train_FMRI,val_FMRI,\
                        test_FMRI,train_labels,val_labels, test_labels,train_FMRI_averaged_all,val_FMRI_averaged_all,test_FMRI_averaged_all
        if (len(shared_all_indexes) != 0 ):
                return train_images,val_images,test_images,train_FMRI,val_FMRI,\
                test_FMRI,train_labels,val_labels, test_labels,test_FMRI_shared_across,test_images_shared_across
        else:
                return train_images,val_images,test_images,train_FMRI,val_FMRI,\
                        test_FMRI,train_labels,val_labels, test_labels

def Split_data(Images_cropped_data,FMRI_data,Image_labels,averaged_data_flag,shared_mask,Shared_mask_Averaged,including_ind_mask):
        # Leave out the same 100 of the 1000 shared images from any subject
        # Take out another 900 from the unique images. 
        if(averaged_data_flag == 1):
                train_indexes,val_indexes, test_indexes =Train_Test_Valdiation_Split_10k_ordered(FMRI_data.shape[0],Shared_mask_Averaged,\
                        Image_labels,including_ind_mask = including_ind_mask)
        else:
                train_indexes,val_indexes, test_indexes = Train_Test_Valdiation_Split(FMRI_data.shape[0],shared_mask)
        train_images = Images_cropped_data[train_indexes[0],:,:,:]
        train_images = np.swapaxes(np.swapaxes(train_images,1,3),2,3)
        val_images = Images_cropped_data[val_indexes[0],:,:,:]
        val_images = np.swapaxes(np.swapaxes(val_images,1,3),2,3)
        test_images = Images_cropped_data[test_indexes[0],:,:,:]
        test_images = np.swapaxes(np.swapaxes(test_images,1,3),2,3)
        train_FMRI = FMRI_data[train_indexes[0],:]
        val_FMRI = FMRI_data[val_indexes[0],:]
        test_FMRI = FMRI_data[test_indexes[0],:]
        # train_FMRI_averaged = FMRI_Averaged[train_indexes[0],:]
        # val_FMRI_averaged = FMRI_Averaged[val_indexes[0],:]
        # test_FMRI_averaged = FMRI_Averaged[test_indexes[0],:]
        train_labels = Image_labels[0,train_indexes[0]]
        val_labels = Image_labels[0,val_indexes[0]]
        test_labels = Image_labels[0,test_indexes[0]]
        # switch dimension 1 and 3
        return train_images,val_images,test_images,train_FMRI,val_FMRI,\
                test_FMRI,train_labels,val_labels, test_labels

def Split_data_shared(Images_cropped_data,FMRI_data,Image_labels,averaged_data_flag,shared_mask,Shared_mask_Averaged,including_ind_mask = 0):
        # Leave out the same 100 of the 1000 shared images from any subject
        # Take out another 900 from the unique images. 
        if(averaged_data_flag == 1):
                train_indexes,val_indexes, test_indexes =Train_Test_Valdiation_Split_10k_ordered(FMRI_data.shape[0],Shared_mask_Averaged,\
                        shared_only = 1,including_ind_mask = including_ind_mask)

        train_images = Images_cropped_data[train_indexes[0],:,:,:]
        train_images = np.swapaxes(np.swapaxes(train_images,1,3),2,3)
        val_images = Images_cropped_data[val_indexes[0],:,:,:]
        val_images = np.swapaxes(np.swapaxes(val_images,1,3),2,3)
        test_images = Images_cropped_data[test_indexes[0],:,:,:]
        test_images = np.swapaxes(np.swapaxes(test_images,1,3),2,3)
        train_FMRI = FMRI_data[train_indexes[0],:]
        val_FMRI = FMRI_data[val_indexes[0],:]
        test_FMRI = FMRI_data[test_indexes[0],:]
        train_labels = Image_labels[0,train_indexes[0]]
        val_labels = Image_labels[0,val_indexes[0]]
        test_labels = Image_labels[0,test_indexes[0]]
        # switch dimension 1 and 3
        return train_images,val_images,test_images,train_FMRI,val_FMRI,\
                test_FMRI,train_labels,val_labels, test_labels

def Split_data_clasification(Images_cropped_data,FMRI_data,Image_labels,Images_Coco_labels,\
        Images_Coco_labels_Averaged,Images_Coco_super_label,averaged_data_flag):
        # Leave out the same 100 of the 1000 shared images from any subject
        # Take out another 900 from the unique images. 
        if(averaged_data_flag == 1):
                train_indexes,val_indexes, test_indexes =Train_Test_Valdiation_Split_10k_ordered(FMRI_data.shape[0])
        else:
                train_indexes,val_indexes, test_indexes = Train_Test_Valdiation_Split(FMRI_data.shape[0])
        train_images = Images_cropped_data[train_indexes[0],:,:,:]
        train_images = np.swapaxes(np.swapaxes(train_images,1,3),2,3)
        val_images = Images_cropped_data[val_indexes[0],:,:,:]
        val_images = np.swapaxes(np.swapaxes(val_images,1,3),2,3)
        test_images = Images_cropped_data[test_indexes[0],:,:,:]
        test_images = np.swapaxes(np.swapaxes(test_images,1,3),2,3)
        train_FMRI = FMRI_data[train_indexes[0],:]
        val_FMRI = FMRI_data[val_indexes[0],:]
        test_FMRI = FMRI_data[test_indexes[0],:]
        train_Coco_labels = Images_Coco_labels[train_indexes[0]]
        val_Coco_labels = Images_Coco_labels[val_indexes[0]]
        test_Coco_labels = Images_Coco_labels[test_indexes[0]]
        train_labels = Image_labels[0,train_indexes[0]]
        val_labels = Image_labels[0,val_indexes[0]]
        test_labels = Image_labels[0,test_indexes[0]]
        train_super_label = Images_Coco_super_label[train_indexes[0]]
        val_super_label = Images_Coco_super_label[val_indexes[0]]
        test_super_label = Images_Coco_super_label[test_indexes[0]]

        if(averaged_data_flag == 1):
                train_Coco_labels_Averaged = Images_Coco_labels_Averaged[train_indexes[0]]
                val_Coco_labels_Averaged = Images_Coco_labels_Averaged[val_indexes[0]]
                test_Coco_labels_Averaged = Images_Coco_labels_Averaged[test_indexes[0]]
                return train_images.float(),val_images.float(),test_images.float(),train_FMRI.float(),val_FMRI.float(),\
                        test_FMRI.float(),train_labels,val_labels, test_labels,train_Coco_labels, val_Coco_labels, test_Coco_labels,\
                        train_Coco_labels_Averaged,val_Coco_labels_Averaged,test_Coco_labels_Averaged,train_super_label,val_super_label,test_super_label
        else:
                return train_images.float(),val_images.float(),test_images.float(),train_FMRI.float(),val_FMRI.float(),\
                        test_FMRI.float(),train_labels,val_labels, test_labels,train_Coco_labels, val_Coco_labels,\
                                 test_Coco_labels,train_super_label,val_super_label,test_super_label
                
def Train_Test_Split(exp_len):
        # Leave out the same 100 of the 1000 shared images from any subject (already sorted the same for all sub)
        # Take out another 900 from the unique images.
        stim_info, exp_info = get_FMRI_info()
        subj_img_seuqence_10k_ids = exp_info['masterordering']
        test_image_10k_mask = np.zeros(10000)
        test_image_10k_mask[:100] = 1
        test_image_10k_mask[1000:1900] = 1
        train_image_10k_mask = 1 - test_image_10k_mask
        train_indexes = np.nonzero(train_image_10k_mask[subj_img_seuqence_10k_ids[0,:exp_len]-1])
        test_indexes = np.nonzero(test_image_10k_mask[subj_img_seuqence_10k_ids[0,:exp_len]-1])
        return train_indexes, test_indexes


def Train_Test_Valdiation_Split(exp_len,shared_mask):
        # Leave out the same 100 of the 1000 shared images from any subject (already sorted the same for all sub)
        # Take out another 900 from the unique images.
        stim_info, exp_info = get_FMRI_info()
        subj_img_seuqence_10k_ids = exp_info['masterordering']
        test_image_10k_mask = np.zeros(10000)
        val_image_10k_mask = np.zeros(10000)
        test_image_10k_mask[:100] = 1
        test_image_10k_mask[1000:1900] = 1
        val_image_10k_mask[100:200] = 1
        val_image_10k_mask[1900:2800] = 1
        train_image_10k_mask = 1 - test_image_10k_mask - val_image_10k_mask
        train_indexes = np.nonzero(train_image_10k_mask[subj_img_seuqence_10k_ids[0,:exp_len]-1])
        val_indexes = np.nonzero(val_image_10k_mask[subj_img_seuqence_10k_ids[0,:exp_len]-1])
        test_indexes = np.nonzero(test_image_10k_mask[subj_img_seuqence_10k_ids[0,:exp_len]-1])
        return train_indexes,val_indexes, test_indexes

def Train_Test_Valdiation_Split_10k_ordered(exp_len,shared_mask,Image_labels = 0,shared_only = 0,including_ind_mask = 0):
        # Leave out the same 100 of the 1000 shared images from any subject (already sorted the same for all sub)
        # Take out another 900 from the unique images.
        test_image_10k_mask = np.zeros(exp_len)
        val_image_10k_mask = np.zeros(exp_len)
        train_image_10k_mask= np.zeros(exp_len)
        non_shared_mask = 1 - shared_mask
        shared_indexes = np.nonzero(shared_mask[:,0])
        non_shared_indexes = np.nonzero(non_shared_mask[:,0])

        # Image_labels_shared = Image_labels[0,shared_indexes]
        # Image_labels_shared_ind_sorted = np.argsort(Image_labels_shared[:,0])
        # shared_indexes_sorted  = shared_indexes[Image_labels_shared_ind_sorted]

        
        test_image_10k_mask[shared_indexes[np.nonzero(including_ind_mask < 100)]] = 1
        val_image_10k_mask[shared_indexes[np.nonzero((including_ind_mask >= 100) * (including_ind_mask < 200))]] = 1
        if(shared_only == 0):
                test_image_10k_mask[non_shared_indexes[:900]] = 1
                val_image_10k_mask[non_shared_indexes[900:1800]] = 1
                train_image_10k_mask = 1 - test_image_10k_mask - val_image_10k_mask
        else:
              train_image_10k_mask[shared_indexes[np.nonzero((including_ind_mask >= 200) * (including_ind_mask < 1000))]]   = 1
        train_indexes = np.nonzero(train_image_10k_mask)
        val_indexes = np.nonzero(val_image_10k_mask)
        test_indexes = np.nonzero(test_image_10k_mask)
        return train_indexes,val_indexes, test_indexes

def Split_data_shared_and_nonshared(Images,FMRI,labels,including_ind_s,including_ind_ns):
        Images_s = Images[including_ind_s]
        Images_ns = Images[including_ind_ns]
        FMRI_s = FMRI[including_ind_s]
        FMRI_ns = FMRI[including_ind_ns]
        labels_s = labels[0,including_ind_s]
        labels_ns = labels[0,including_ind_ns]

        train_images_ns = Images_ns[:-1000]
        val_images_ns = Images_ns[-1000:-500]
        test_images_ns = Images_ns[-500:]
        train_FMRI_ns = FMRI_ns[:-1000]
        val_FMRI_ns = FMRI_ns[-1000:-500]
        test_FMRI_ns = FMRI_ns[-500:]
        train_labels_ns = labels_ns[:-1000]
        val_labels_ns = labels_ns[-1000:-500]
        test_labels_ns = labels_ns[-500:]

        train_images_s = Images_s[:-100]
        val_images_s = Images_s[-100:-50]
        test_images_s = Images_s[-50:]
        train_FMRI_s = FMRI_s[:-100]
        val_FMRI_s = FMRI_s[-100:-50]
        test_FMRI_s = FMRI_s[-50:]
        train_labels_s = labels_s[:-100]
        val_labels_s = labels_s[-100:-50]
        test_labels_s = labels_s[-50:]        

        # switch dimension 1 and 3
        return train_images_s,val_images_s,test_images_s,train_FMRI_s,val_FMRI_s,\
                test_FMRI_s,train_labels_s,val_labels_s, test_labels_s,\
                train_images_ns,val_images_ns,test_images_ns,train_FMRI_ns,val_FMRI_ns,\
                test_FMRI_ns,train_labels_ns,val_labels_ns, test_labels_ns



def get_FMRI_info():
        """ Images info
        Column 1 is the 0-based image number (0-72999).
        Column 2 (cocoId) is the ID number assigned to this image in the COCO database.
        Column 3 (cocoSplit) is either “train2017” or “val2017”. The COCO web site designates different splits of images into training and validation sets. The NSD experiment does not involve any use of this designation (such as in the experimental design), but we provide this information just in case it is useful.
        Column 4 (cropBox) is a tuple of four numbers indicating how the original COCO image was cropped. The format is (top, bottom, left, right) in fractions of image size. Notice that cropping was always performed along only the largest dimension. Thus, there are always two 0’s in the cropBox.
        Column 5 (loss) is the object-loss score after cropping. See manuscript for more details, as well as the "Details on crop selection for COCO images" section below.
        Column 6 (nsdId) is the 0-based index of the image into the full set of 73k images used in the NSD experiment. Values are the same as column 1.
        Column 7 (flagged) is True if the image has questionable content (e.g. violent or salacious content).
        Column 8 (BOLD5000) is True if the image is included in the BOLD5000 dataset (http://bold5000.github.io). Note that NSD images are square-cropped, so the images are not quite identical across the two datasets.
        Column 9 (shared1000) is True if the image is one of the special 1,000 images that are shown to all 8 subjects in the NSD experiment.
        Columns 10-17 (subjectX) is 0 or 1 indicating whether that image was shown to subjectX (X ranges from 1-8).
        Columns 18-41 (subjectX_repN) is 0 indicating that the image was not shown to subjectX, or a positive integer T indicating that the image was shown to subjectX on repetitionN (X ranges from 1-8; N ranges from 0-2 for a total of 3 trials). T provides the trialID associated with the image showing. The trialID is a 1-based index from 1 to 30000 corresponding to the chronological order of all 30,000 stimulus trials that a subject encounters over the course of the NSD experiment. Each of the 73k NSD images either has 3 trialIDs (if it was shown to only one subject) or 24 trialIDs (if it was shown to all 8 subjects). """
        """FMRI Info 
        Contents:
        <masterordering> is 1 x 30000 with the sequence of trials (indices relative to 10k)
        <basiccnt> is 3 x 40 where we calculate, for each scan session separately, the number of distinct images in that session that have a number of presentations equal to the row index.
        <sharedix> is 1 x 1000 with sorted indices of the shared images (relative to 73k)
        <subjectim> is 8 x 10000 with indices of images (relative to 73k). the first 1000 are the common shared 1000 images. it turns out that the indices for these 1000 are in sorted order. this is for simplicity, and there is no significance to the order (since the order in which the 1000 images are shown is randomly determined). the remaining 9000 for each subject are in a randomized non-sorted order.
        <stimpattern> is 40 sessions x 12 runs x 75 trials. elements are 0/1 indicating when stimulus trials actually occur. note that the same <stimpattern> is used for all subjects.
        Note: subjectim(:,masterordering) is 8 x 30000 indicating the temporal sequence of 73k-ids shown to each subject. This sequence refers only to the stimulus trials (ignoring the blank trials and the rest periods at the beginning and end of each run)."""
        
        # Load stimulus info
        data_info_dir = '/home/navvew/data/NSD/nsddata/experiments/nsd/'
        data_stim_info_file = 'nsd_stim_info_merged.csv'
        data_stim_info = open(data_info_dir + data_stim_info_file)
        csvreader_stim_info = csv.reader(data_stim_info)
        header = []
        header = next(csvreader_stim_info)
        stim_info = []
        for row in csvreader_stim_info:
                stim_info.append(row)
        data_stim_info.close()

        # Load expriments design
        data_exp_info_file = 'nsd_expdesign.mat'
        exp_info = scipy.io.loadmat(data_info_dir + data_exp_info_file)
        return np.array(stim_info), exp_info

def get_ROI_data(roi_dir_old,subj_num,func_type):
        # Load the roi maps and create dictionaries:
        # roi_map (6,voxels 3D), 0 dienstion for diffrent map type
        # roi_label_map (6,dict), first dimenstion for diffrent map type, each row is diffrent dict 
        # roi_type_dict (6), dictionary for each 0 dimension in roi_map the map type
        roi_label_map = {}
        roi_dir = roi_dir_old + 'subj0' + str(subj_num) + '/' + func_type + '/roi/'
        f1 = nib.load(roi_dir+ 'prf-eccrois.nii.gz')
        roi_eccrois = torch.tensor(f1.get_data(),dtype = torch.float32)
        roi_map = torch.zeros(6,roi_eccrois.shape[0],roi_eccrois.shape[1],roi_eccrois.shape[2]) - torch.tensor(1)
        roi_map[0,:,:,:] = roi_eccrois
        roi_label_map[0] = {0 : 'Unknown', 1 : 'ecc0pt5', 2 : 'ecc1', 3 : 'ecc2', 4 : 'ecc4', 5 : 'ecc4+'}
        f2 = nib.load(roi_dir+ 'prf-visualrois.nii.gz')
        roi_map[1,:,:,:] = torch.tensor(f2.get_data(),dtype = torch.float32)
        roi_label_map[1] = { 1 : 'V1v', 2 : 'V1d', 3 : 'V2v', 4 : 'V2d', 5 : 'V3v', 6 : 'V3d', 7 : 'hV4'}
        f3 = nib.load(roi_dir+ 'floc-faces.nii.gz')
        roi_map[2,:,:,:] = torch.tensor(f3.get_data(),dtype = torch.float32)
        roi_label_map[2] = { 1 : 'OFA', 2 : 'FFA-1', 3 : 'FFA-2', 4 : 'mTL-faces', 5 : 'aTL-faces'}
        f4 = nib.load(roi_dir+ 'floc-bodies.nii.gz')
        roi_map[3,:,:,:] = torch.tensor(f4.get_data(),dtype = torch.float32)
        roi_label_map[3] = { 1 : 'EBA', 2 : 'FBA-1', 3 : 'FBA-2', 4 : 'mTL-bodies'}
        f5 = nib.load(roi_dir+ 'floc-places.nii.gz')
        roi_map[4,:,:,:] = torch.tensor(f5.get_data(),dtype = torch.float32)
        roi_label_map[4] = { 1 : 'OPA', 2 : 'PPA', 3 : 'RSC'}
        f6 = nib.load(roi_dir+ 'floc-words.nii.gz')
        roi_map[5,:,:,:] = torch.tensor(f6.get_data(),dtype = torch.float32)
        roi_label_map[5] = { 1 : 'OWFA', 2 : 'VWFA-1', 3 : 'VWFA-2', 4 : 'mfs-words', 5 : 'mTL-words'}
        roi_type_dict = {}
        roi_type_dict[0] = 'eccrois'
        roi_type_dict[1] = 'visualrois'
        roi_type_dict[2] = 'faces'
        roi_type_dict[3] = 'bodies'
        roi_type_dict[4] = 'places'
        roi_type_dict[5] = 'words'

        return roi_map, roi_label_map,roi_type_dict

def get_ROI_Visual_Cortex(roi_dir_old,subj_num,func_type,FMRI_voxels_cordinates,Include_HVC = False):
        roi_map, roi_label_map,roi_type_dict = get_ROI_data(roi_dir, subj_num,func_type)
        visual_cortex = roi_map[1,:,:,:]
        ROI_V1_map = (visual_cortex == 1) | (visual_cortex == 2)
        ROI_V2_map = (visual_cortex == 3) | (visual_cortex == 4)
        ROI_V3_map = (visual_cortex == 5) | (visual_cortex == 6)
        ROI_V4_map = (visual_cortex == 7)
        ROI_V1_flat = ROI_V1_map[FMRI_voxels_cordinates[:,0],FMRI_voxels_cordinates[:,1],FMRI_voxels_cordinates[:,2]]
        ROI_V2_flat = ROI_V2_map[FMRI_voxels_cordinates[:,0],FMRI_voxels_cordinates[:,1],FMRI_voxels_cordinates[:,2]]
        ROI_V3_flat = ROI_V3_map[FMRI_voxels_cordinates[:,0],FMRI_voxels_cordinates[:,1],FMRI_voxels_cordinates[:,2]]
        ROI_V4_flat = ROI_V4_map[FMRI_voxels_cordinates[:,0],FMRI_voxels_cordinates[:,1],FMRI_voxels_cordinates[:,2]]
        if(Include_HVC):
                ROI_faces_map = roi_map[2,:,:,:] > 0 
                ROI_bodies_map = roi_map[3,:,:,:] > 0 
                ROI_places_map = roi_map[4,:,:,:] > 0 
                ROI_words_map = roi_map[5,:,:,:] > 0 
                ROI_faces_flat = ROI_faces_map[FMRI_voxels_cordinates[:,0],FMRI_voxels_cordinates[:,1],FMRI_voxels_cordinates[:,2]]
                ROI_bodies_flat = ROI_bodies_map[FMRI_voxels_cordinates[:,0],FMRI_voxels_cordinates[:,1],FMRI_voxels_cordinates[:,2]]
                ROI_places_flat = ROI_places_map[FMRI_voxels_cordinates[:,0],FMRI_voxels_cordinates[:,1],FMRI_voxels_cordinates[:,2]]
                ROI_words_flat = ROI_words_map[FMRI_voxels_cordinates[:,0],FMRI_voxels_cordinates[:,1],FMRI_voxels_cordinates[:,2]]

                return ROI_V1_map,ROI_V2_map,ROI_V3_map,ROI_V4_map,ROI_V1_flat,ROI_V2_flat,ROI_V3_flat,ROI_V4_flat,\
                        ROI_faces_map,ROI_bodies_map,ROI_places_map,ROI_words_map,ROI_faces_flat,ROI_bodies_flat,\
                             ROI_places_flat,ROI_words_flat 
        else:
                return ROI_V1_map,ROI_V2_map,ROI_V3_map,ROI_V4_map,ROI_V1_flat,ROI_V2_flat,ROI_V3_flat,ROI_V4_flat


def  NC_from_NCSNR(NCSNR,n):
        NC = 100*(NCSNR**2 /(NCSNR**2 + 1/n))
        return NC