from Utils.gen_functions import calc_snr
from Utils.vim1_utils import *
import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn import preprocessing



class vim1_data_handler():
    """Generate batches for FMRI prediction
    frames_back - how many video_self frames to take before FMRI frame
    frames_forward - how many video_self frames to take after FMRI frame
    """

    def __init__(self, subject = 1 , norm = 0 ,select_areas =None ,select_by_snr = 1,num_voxels = 8000,log= False):  #normalize 0-no, 1-seperate, 2-together

        data = get_processed_data(subject)
        self.train = np.nan_to_num(data[0])
        self.test  = np.nan_to_num(data[1])
        self.train_label  = data[2]
        self.test_label  = data[3]
        self.norm = norm
        self.select_by_snr = select_by_snr
        self.roi = get_roi_label(subject)[subject][:,0]
        self.select_areas = select_areas
        self.num_voxels = num_voxels
        self.log = log


    def get_data(self):

        train = self.train
        test = self.test
        num_vox = test.shape[1]

        if(self.log):
            train = np.log(1+np.abs(train))*np.sign(train)
            test  = np.log(1+np.abs(test))*np.sign(test)


        if(self.select_areas is not None):
            select = np.zeros(num_vox,dtype=bool)
            for v in self.select_areas:
                select = select| (v ==self.roi)
            train = train[:,select]
            test  = test[:,select]
            num_vox = np.sum(select)


        if(self.norm == 1):
            train = sklearn.preprocessing.scale(train)
            test = sklearn.preprocessing.scale(test)
        if(self.norm == 2):
            num_train = train.shape[0]
            full = np.concatenate([train,test],axis=0)
            full = sklearn.preprocessing.scale(full)
            train = full[:num_train,]
            test = full[num_train:, ]

        test_avg = np.zeros([120,num_vox])


        for i in range(120):
            test_avg[i] = np.mean(test[self.test_label==i],axis=0)

        train_avg = np.zeros([1750, num_vox])

        for i in range(1750):
            train_avg[i] = np.mean(train[self.train_label == i], axis=0)

        snr = np.nan_to_num(calc_snr(test,test_avg,self.test_label))

        if(self.select_by_snr):
            snr_cp = np.copy(snr)
            snr_cp.sort()
            snr_cp = snr_cp[::-1]
            th = snr_cp[self.num_voxels]
            select = (snr>th)
            train = train[:,select]
            test  = test[:,select]
            train_avg = train_avg[:, select]
            test_avg = test_avg[:, select]
            snr  = snr[select]

        train_avg_exp = np.expand_dims(train_avg, axis=1)
        train_exp = np.expand_dims(train, axis=1)
        train_re = np.concatenate([train_exp[:1750], train_exp[1750:], train_avg_exp,train_avg_exp], axis=1)
        dict_ = {}
        dict_['snr'] = snr
        dict_['train_re'] = train_re
        dict_['test'] = test
        dict_['test_avg'] = test_avg
        dict_['train_avg'] = train_avg
        dict_['test_label'] = self.test_label

        return dict_

handler = vim1_data_handler()
