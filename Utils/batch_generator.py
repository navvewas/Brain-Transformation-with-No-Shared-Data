
from keras.utils import Sequence
from keras.backend import random_normal_variable
import numpy as np
import os
import random
from scipy.misc import imread
import pandas as pd
# from torch import dtype
from Utils.image_functions import image_prepare, rand_shift
import pickle
import tensorflow as tf
from keras import layers
import scipy

class batch_generator_2_subj(Sequence):
    def __init__(self, X_1, Y_1,X_2,Y_2, batch_size=32, labels = None):
        self.indexes = {}
        self.batch_size = batch_size
        self.Y_1 = Y_1
        self.Y_2 = Y_2
        self.X_1= X_1
        self.X_2= X_2

        self.labels = labels
        self.indexes  = np.random.permutation(self.Y_1.shape[0])
    def __len__(self):
        'Denotes the number of batches per epoch'
        len = max(int(self.Y_1.shape[0] // self.batch_size), 1)
        return len
    def __getitem__(self,batch_num):
        indexes = (self.indexes)[batch_num * self.batch_size:(batch_num + 1) * self.batch_size]
        return self.Y_1[indexes],self.Y_2[indexes], self.X_1[indexes],self.X_2[indexes]
    def on_epoch_end(self):
        self.indexes = np.random.permutation(self.indexes)

class batch_generator_3_subj_shared(Sequence):
    def __init__(self, X_1, Y_1,X_2,Y_2,X_3,Y_3, batch_size=32, labels = None):
        self.indexes = {}
        self.batch_size = batch_size
        self.Y_1 = Y_1
        self.Y_2 = Y_2
        self.Y_3 = Y_3
        self.X_1= X_1
        self.X_2= X_2
        self.X_3= X_3
        self.labels = labels
        self.min_size = np.minimum(np.minimum(Y_1.shape[0],Y_2.shape[0]),Y_3.shape[0])
        self.indexes  = np.random.permutation(Y_1.shape[0])
    def __len__(self):
        'Denotes the number of batches per epoch'
        len = max(int(self.min_size // self.batch_size), 1)
        return len
    def __getitem__(self,batch_num):
        indexes = (self.indexes)[batch_num * self.batch_size:(batch_num + 1) * self.batch_size]

        return self.Y_1[indexes],self.Y_2[indexes],self.Y_3[indexes], self.X_1[indexes],self.X_2[indexes],self.X_3[indexes]
    def on_epoch_end(self):
        self.indexes = np.random.permutation(self.indexes)

class batch_generator_3_subj(Sequence):
    def __init__(self, X_1, Y_1,X_2,Y_2,X_3,Y_3, batch_size=32, labels = None):
        self.indexes = {}
        self.batch_size = batch_size
        self.Y_1 = Y_1
        self.Y_2 = Y_2
        self.Y_3 = Y_3
        self.X_1= X_1
        self.X_2= X_2
        self.X_3= X_3
        self.labels = labels
        self.min_size = np.minimum(np.minimum(Y_1.shape[0],Y_2.shape[0]),Y_3.shape[0])
        self.indexes_1  = np.random.permutation(Y_1.shape[0])
        self.indexes_2  = np.random.permutation(Y_2.shape[0])
        self.indexes_3  = np.random.permutation(Y_3.shape[0])
    def __len__(self):
        'Denotes the number of batches per epoch'
        len = max(int(self.min_size // self.batch_size), 1)
        return len
    def __getitem__(self,batch_num):
        indexes_1 = (self.indexes_1)[batch_num * self.batch_size:(batch_num + 1) * self.batch_size]
        indexes_2 = (self.indexes_2)[batch_num * self.batch_size:(batch_num + 1) * self.batch_size]
        indexes_3 = (self.indexes_3)[batch_num * self.batch_size:(batch_num + 1) * self.batch_size]

        return self.Y_1[indexes_1],self.Y_2[indexes_2],self.Y_3[indexes_3], self.X_1[indexes_1],self.X_2[indexes_2],self.X_3[indexes_3]
    def on_epoch_end(self):
        self.indexes_1 = np.random.permutation(self.indexes_1)
        self.indexes_2 = np.random.permutation(self.indexes_2)
        self.indexes_3 = np.random.permutation(self.indexes_3)


class batch_generator_dec(Sequence):

    def __init__(self,  X, Y, batch_size =32, labels = None):
        self.indexes = {}
        self.batch_size = batch_size
        self.Y = Y
        self.X = X
        self.labels = labels
        if self.labels is None:
            self.frac = 1
        else:
            self.frac = (self.labels == 1).sum() #Find the number of fmris per image

        self.indexes  = np.random.permutation(self.Y.shape[0]//self.frac)

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.batch_size == 12:
            return 75
        len = max(int(self.Y.shape[0] // self.batch_size), 1)
        return len//(self.frac)

    def __getitem__(self,batch_num):
        indexes = (self.indexes)[batch_num * self.batch_size:(batch_num + 1) * self.batch_size]
        if self.frac == 1:
            return self.Y[indexes], self.X[indexes]
        else:
            y = np.zeros((self.batch_size, self.Y.shape[1]))
            x = np.zeros((self.batch_size, self.X.shape[1], self.X.shape[2], self.X.shape[3]))
            for i in range(self.batch_size):
                all_fmris = self.Y[self.labels==indexes[i]]
                num_samples = np.random.randint(3,6)
                samples = np.random.permutation(self.frac)[0:num_samples]  # choose 3 random samples
                chosen_fmris = all_fmris[samples]
                y[i] = np.mean(chosen_fmris, axis=0, keepdims=True)
                x[i] = self.X[self.labels==indexes[i]][0]
            return y, x

    def on_epoch_end(self):
        self.indexes = np.random.permutation(self.indexes)

class batch_generator_dec_with_avg(Sequence):

    def __init__(self,  X, Y,Y_avg, batch_size =32, labels = None):
        self.indexes = {}
        self.batch_size = batch_size
        self.Y = Y
        self.X = X
        self.Y_avg = Y_avg
        self.labels = labels
        self.indexes  = np.random.permutation(self.Y.shape[0])

    def __len__(self):
        'Denotes the number of batches per epoch'
        len = max(int(self.Y.shape[0] // self.batch_size), 1)
        return len

    def __getitem__(self,batch_num):
        indexes = (self.indexes)[batch_num * self.batch_size:(batch_num + 1) * self.batch_size]
        return self.Y[indexes], self.X[indexes],self.Y_avg[indexes]


    def on_epoch_end(self):
        self.indexes = np.random.permutation(self.indexes)

class batch_generator_enc(batch_generator_dec):
    def __init__(self, X, Y, batch_size=32,max_shift = 5, labels = None):
        super().__init__(X, Y, batch_size, labels)
        self.max_shift = max_shift

    def __getitem__(self,batch_num, labels = None):
        y, x = super().__getitem__(batch_num)
        x_shifted = np.zeros(x.shape)
        for i in range(x.shape[0]):
            x_shifted[i] = rand_shift(x[i],self.max_shift)
        return x_shifted,y





class batch_generator_enc_multi(batch_generator_3_subj):
    def __init__(self, X_1, Y_1,X_2,Y_2,X_3,Y_3, batch_size=32,max_shift = 5, labels = None):
        super().__init__(X_1, Y_1,X_2,Y_2,X_3,Y_3, batch_size, labels)
        self.max_shift = max_shift

    def __getitem__(self,batch_num, labels = None):
        y_1, y_2,y_3, x_1,x_2,x_3 = super().__getitem__(batch_num)
        x_shifted_1 = np.zeros(x_1.shape)
        for i in range(x_1.shape[0]):
            x_shifted_1[i] = rand_shift(x_1[i],self.max_shift)
        x_shifted_2 = np.zeros(x_2.shape)
        for i in range(x_2.shape[0]):
            x_shifted_2[i] = rand_shift(x_2[i],self.max_shift)
        x_shifted_3 = np.zeros(x_3.shape)
        for i in range(x_3.shape[0]):
            x_shifted_3[i] = rand_shift(x_3[i],self.max_shift)
        return [y_1,y_2,y_3,x_shifted_1,x_shifted_2,x_shifted_3],[y_1,y_2,y_3,y_1,y_1,y_1,y_1,y_1,y_1,y_1,y_1,y_1]

class batch_generator_enc_self_augm(Sequence):
    def __init__(self, X, Y, train_labels = None, batch_size = 32, batch_unpaired = 32, max_shift = 5, img_len = 112
                 ,ext_dir =  '/net/mraid11/export/data/navvew/SSReconstnClass/data/ImageNet_Files/train_images', num_ext_per_class=150):
        self.gen_enc        = batch_generator_enc(X, Y, batch_size=batch_size,max_shift = max_shift, labels = train_labels)
        self.gen_ext_im     = batch_generator_external_images(img_size = img_len, batch_size=batch_unpaired,ext_dir = ext_dir, num_ext_per_class=num_ext_per_class)
    def on_epoch_end(self):
        self.gen_enc.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return 1
    def __getitem__(self,batch_num):
        x_in, y_out = self.gen_enc.__getitem__(batch_num)
        x_ext_in, x_ext_out = self.gen_ext_im.__getitem__(batch_num)

        return [x_in, x_ext_in], [y_out, y_out,y_out,y_out,y_out,y_out,y_out]


class batch_generator_enc_self_corr(Sequence):
    def __init__(self, X, Y,corr_mean, train_labels = None, batch_size = 32, batch_unpaired = 32, max_shift = 5, img_len = 112
                 ,ext_dir =  '/net/mraid11/export/data/navvew/SSReconstnClass/data/ImageNet_Files/train_images', num_ext_per_class=150,use_saved = 0,augm_and_not_ext = 0):
        self.gen_enc        = batch_generator_enc(X, Y, batch_size=batch_size,max_shift = max_shift, labels = train_labels)
        self.gen_ext_im     = batch_generator_external_images(img_size = img_len, batch_size=batch_unpaired,ext_dir = ext_dir, num_ext_per_class=num_ext_per_class,use_saved =use_saved )
        self.corr_mean = corr_mean
        self.augm_and_not_ext = augm_and_not_ext
    def on_epoch_end(self):
        self.gen_enc.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return 1
    def __getitem__(self,batch_num):
        x_in, y_out = self.gen_enc.__getitem__(batch_num)
        x_ext_in, x_ext_out = self.gen_ext_im.__getitem__(batch_num)

        if(self.augm_and_not_ext == 0):
            return [x_in, x_ext_in], [y_out, y_out,y_out]
        else:
            choose_rtation = np.round(np.random.rand(1) * 2)
            x_augm = scipy.ndimage.rotate(x_in,choose_rtation * 90 + 90,[2,1])
            # diff = x_augm_tmp.shape[1] - x_in.shape[1]
            # diff_1 = np.int(diff/2)
            # diff_2 = diff - diff_1
            # if(diff_2 == 0):
            #     x_augm = x_augm_tmp[:,diff_1:,diff_1:,:]
            # else:
            #     x_augm = x_augm_tmp[:,diff_1:-diff_2,diff_1:-diff_2,:]
            return [x_in, x_augm], [y_out, y_out,y_out]

class batch_generator_external_images(Sequence):
    """
    Gets images from an image directory
    """
    def __init__(self, img_size = 112, batch_size=16,ext_dir = '/net/mraid11/export/data/navvew/SSReconstnClass/data/ImageNet_Files/train_images',\
         num_ext_per_class=150,use_saved =0, to_grey =0):
        self.img_size = img_size
        self.batch_size = batch_size
        files = os.listdir(ext_dir)
        self.img_files = []
        self.test_im = pd.read_csv('/net/mraid11/export/data/navvew/SSReconstnClass/data/Kamitani_Files/images/imageID_test.csv', header=None)
        self.to_grey = to_grey
        self.num_ext_per_class = num_ext_per_class
        self.save_file_dir = '/net/mraid11/export/data/navvew/SSReconstnClass/data/Kamitani_Files/images/ext_images_files'
        if(use_saved == 1):
            with open(self.save_file_dir, "rb") as fp:   # Unpickling
                self.img_files = pickle.load(fp)
            self.ext_dir = ext_dir
        else:
            for file in files:
                img_files = []
                if os.path.isdir(ext_dir + '/' + file) and file.startswith('n'):
                    img_files = random.sample(os.listdir(ext_dir + '/' + file), self.num_ext_per_class)
                    for i in range(img_files.__len__()):
                        img_files[i] = file + '/' + img_files[i]
                    self.img_files += img_files
                elif file.endswith("JPEG"):
                    self.img_files.append(file)
            self.ext_dir = ext_dir
            # print('ext_dir: ' + ext_dir + ' img_files len: ' + str(self.img_files.__len__()))
            with open(self.save_file_dir, "wb") as fp:
                pickle.dump(self.img_files, fp)

    def __getitem__(self, batch_num):
        img_file = random.sample(self.img_files, self.batch_size)

        images_in = np.zeros([self.batch_size, self.img_size, self.img_size, 3])
        images_out = np.zeros([self.batch_size, self.img_size, self.img_size, 3])
        for i, file in enumerate(img_file):
            img_in, img_out = self.read_file(file)
            images_in[i] = img_in
            images_out[i] = img_out
        if(self.to_grey == 1):
            images_in = np.expand_dims(images_in.mean(-1),-1)
            images_in = np.concatenate([images_in,images_in,images_in],-1)
            images_out = np.expand_dims(images_out.mean(-1),-1)
            images_out = np.concatenate([images_out,images_out,images_out],-1)
        return images_in, images_out

    def read_file(self, file, only_out=False):
        if 'wind' in file:
            img = imread(file)
        else:
            img = imread(self.ext_dir + '/' + file)
        img_in = rand_shift(image_prepare(img, self.img_size), max_shift = 5) if not only_out else None
        img_out = image_prepare(img, self.img_size)
        return (img_in, img_out) if not only_out else img_out

    def __len__(self):
        return  50000// self.batch_size




class batch_generator_test_fmri(Sequence):

    """
    Generates test fMRI samples
    inputs:
        frac - fraction of test fmri to average (3 -> 1/3)
    """
    def __init__(self,Y,labels, batch_size=32, frac =3,ignore_labels = None):
        self.Y = Y
        self.labels = labels
        self.frac = frac
        self.num_vox = Y.shape[1]
        self.batch_size = batch_size
        self.ignore_labels = ignore_labels
        print(self.ignore_labels)

    def __getitem__(self,batch_num):
        y = np.zeros([self.batch_size, self.num_vox])
        for i in range(self.batch_size):
                label = np.random.choice(self.labels, 1)
                if(self.ignore_labels is not None):
                    while(label in self.ignore_labels):
                        label = np.random.choice(self.labels, 1)

                indexes = self.get_random_indexes(label, frac=self.frac)
                y[i] = np.mean(self.Y[indexes, :], axis=0, keepdims=True)

        return y

    def get_random_indexes(self,label,frac =3):
        indexes = np.where(self.labels == label)[0]
        rand_ind = np.random.choice(frac, indexes.shape)
        while (np.sum(rand_ind) == 0 or np.min(rand_ind) > 0):
            rand_ind = np.random.choice(frac, indexes.shape)
        return indexes[rand_ind == 0]#rand_ind


class batch_generator_encdec(Sequence):
    def __init__(self, X, Y, Y_test, test_labels, train_labels = None, batch_paired = 48, batch_unpaired = 16, max_shift_enc = 5, img_len = 112
                 , frac_test = 3,ext_dir =  '/net/mraid11/export/data/navvew/SSReconstnClass/data/ImageNet_Files/train_images', num_ext_per_class=150, ignore_test_fmri_labels = None):
        self.batch_size     = batch_paired+batch_unpaired
        self.gen_dec        = batch_generator_dec(X, Y, batch_size=batch_paired, labels = train_labels)
        self.gen_enc        = batch_generator_enc(X, Y, batch_size=batch_paired,max_shift = max_shift_enc, labels = train_labels)
        self.gen_ext_im     = batch_generator_external_images(img_size = img_len, batch_size=batch_unpaired,ext_dir = ext_dir, num_ext_per_class=num_ext_per_class)
        self.gen_test_fmri  = batch_generator_test_fmri(Y_test,test_labels, batch_size=batch_unpaired, frac =frac_test, ignore_labels = ignore_test_fmri_labels)
        self.batch_paired   = batch_paired
        self.batch_unpaired = batch_unpaired

    def on_epoch_end(self):
        self.gen_dec.on_epoch_end()
        self.gen_enc.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.gen_dec.__len__()
        # return 1
    def __getitem__(self,batch_num):
        y_in, x_out =  self.gen_dec.__getitem__(batch_num)
        x_in, y_out = self.gen_enc.__getitem__(batch_num)
        x_ext_in, x_ext_out = self.gen_ext_im.__getitem__(batch_num)
        y_test_avg = self.gen_test_fmri.__getitem__(batch_num)

        x_in = np.concatenate([x_in, x_ext_in], axis=0)
        x_out = np.concatenate([x_out,x_ext_out], axis=0)
        y_in = np.concatenate([y_in, y_test_avg], axis=0)
        y_out = np.concatenate([y_out, y_test_avg], axis=0)
        mode = np.concatenate([np.ones([self.batch_paired, 1]), np.zeros([self.batch_unpaired, 1])], axis=0)
        return [y_in, x_in, mode], [x_out, y_out]

class batch_generator_encdec_extra_encdec(Sequence):
    def __init__(self, X, Y, Y_test, test_labels, train_labels = None, batch_paired = 48, batch_unpaired = 16, max_shift_enc = 5, img_len = 112
                 , frac_test = 3,ext_dir =  '/net/mraid11/export/data/navvew/SSReconstnClass/data/ImageNet_Files/train_images', num_ext_per_class=150, ignore_test_fmri_labels = None):
        self.batch_size     = batch_paired+batch_unpaired
        self.gen_dec        = batch_generator_dec(X, Y, batch_size=batch_paired, labels = train_labels)
        self.gen_enc        = batch_generator_enc(X, Y, batch_size=batch_paired,max_shift = max_shift_enc, labels = train_labels)
        self.gen_ext_im     = batch_generator_external_images(img_size = img_len, batch_size=batch_unpaired,ext_dir = ext_dir, num_ext_per_class=num_ext_per_class)
        self.gen_test_fmri  = batch_generator_test_fmri(Y_test,test_labels, batch_size=batch_unpaired, frac =frac_test, ignore_labels = ignore_test_fmri_labels)
        self.batch_paired   = batch_paired
        self.batch_unpaired = batch_unpaired

    def on_epoch_end(self):
        self.gen_dec.on_epoch_end()
        self.gen_enc.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.gen_dec.__len__()
        # return 1
    def __getitem__(self,batch_num):
        y_in, x_out =  self.gen_dec.__getitem__(batch_num)
        x_in, y_out = self.gen_enc.__getitem__(batch_num)
        x_ext_in, x_ext_out = self.gen_ext_im.__getitem__(batch_num)
        y_test_avg = self.gen_test_fmri.__getitem__(batch_num)
        return [y_in, x_in, x_ext_in], [x_out,x_ext_out, y_out]


# def pred_dec(model,y):
#     return model.predict( [y,np.zeros([y.shape[0],RESOLUTION,RESOLUTION,3]),np.ones([y.shape[0],1]) ],batch_size=50)[0]
# def pred_enc(model,x):
#     return model.predict( [ np.zeros([x.shape[0],NUM_VOXELS]),x, np.ones([x.shape[0], 1])], batch_size=50)[1]
# def pred_encdec(model,x):
#     return model.predict([np.zeros([x.shape[0],NUM_VOXELS]), x, np.zeros([x.shape[0], 1])], batch_size=50)[0]
class batch_generator_2_subj(Sequence):
    def __init__(self, X_1, X_2,Y_1,Y_2, batch_size=32, labels = None):
        self.indexes = {}
        self.batch_size = batch_size
        self.Y_1 = Y_1
        self.Y_2 = Y_2
        self.X_1= X_1
        self.X_2= X_2

        self.labels = labels
        self.indexes  = np.random.permutation(self.Y_1.shape[0])
    def __getitem__(self,batch_num):
        indexes = (self.indexes)[batch_num * self.batch_size:(batch_num + 1) * self.batch_size]
        return self.Y_1[indexes],self.Y_2[indexes], self.X_1[indexes],self.X_2[indexes]
    def on_epoch_end(self):
        self.indexes = np.random.permutation(self.indexes)
    def __len__(self):
        len = max(int(self.Y_1.shape[0] // self.batch_size), 1)
        return len
class batch_generator_2_subj_with_add_avg(Sequence):
    def __init__(self, X_1, X_2,Y_1,Y_2,Y_1_avg,Y_2_avg, batch_size=32, labels = None):
        self.indexes = {}
        self.batch_size = batch_size
        self.Y_1 = Y_1
        self.Y_2 = Y_2
        self.X_1= X_1
        self.X_2= X_2
        self.Y_1_avg= Y_1_avg
        self.Y_2_avg= Y_2_avg
        self.labels = labels
        self.indexes  = np.random.permutation(self.Y_1.shape[0])
    def __getitem__(self,batch_num):
        indexes = (self.indexes)[batch_num * self.batch_size:(batch_num + 1) * self.batch_size]
        return self.Y_1[indexes],self.Y_2[indexes], self.X_1[indexes],self.X_2[indexes],self.Y_1_avg[indexes],self.Y_2_avg[indexes]
    def on_epoch_end(self):
        self.indexes = np.random.permutation(self.indexes)
    def __len__(self):
        len = max(int(self.Y_1.shape[0] // self.batch_size), 1)
        return len

class batch_generator_2_subj_new(Sequence):
    def __init__(self, X_1, X_2,Y_1,Y_2, batch_size=32, labels = None,max_shift = 5):
        self.indexes = {}
        self.batch_size = batch_size
        self.Y_1 = Y_1
        self.Y_2 = Y_2
        self.X_1= X_1
        self.X_2= X_2
        self.max_shift = max_shift
        self.labels = labels
        self.indexes  = np.random.permutation(self.Y_1.shape[0])
    def __getitem__(self,batch_num):
        indexes = (self.indexes)[batch_num * self.batch_size:(batch_num + 1) * self.batch_size]
        x_1 = self.X_1[indexes]
        x_2 = self.X_2[indexes]
        x1_shifted = np.zeros(x_1.shape)
        for i in range(x_1.shape[0]):
            x1_shifted[i] = rand_shift(x_1[i],self.max_shift)
        x2_shifted = np.zeros(x_2.shape)
        for i in range(x_2.shape[0]):
            x2_shifted[i] = rand_shift(x_2[i],self.max_shift)

        return self.Y_1[indexes],self.Y_2[indexes], x1_shifted,x2_shifted
    def on_epoch_end(self):
        self.indexes = np.random.permutation(self.indexes)
    def __len__(self):
        len = max(int(self.Y_1.shape[0] // self.batch_size), 1)
        return len

class batch_generator_subj_transf(Sequence):
    def __init__(self, X_1, Y_1, X_2, Y_2, train_labels = None, batch_paired = 48, batch_unpaired = 16, max_shift_enc = 5, img_len = 112
                 , frac_test = 3,ext_dir =  '/net/mraid11/export/data/navvew/SSReconstnClass/data/ImageNet_Files/train_images', num_ext_per_class=150, ignore_test_fmri_labels = None):
        self.batch_size     = batch_paired+batch_unpaired
        self.gen_shared        = batch_generator_2_subj(X_1,X_2, Y_1,Y_2, batch_size=batch_paired, labels = train_labels)
        self.gen_ext_im     = batch_generator_external_images(img_size = img_len, batch_size=batch_paired,ext_dir = ext_dir, num_ext_per_class=num_ext_per_class)
        self.batch_paired   = batch_paired
        self.batch_unpaired = batch_unpaired
    def on_epoch_end(self):
        self.gen_shared.on_epoch_end()
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.gen_shared.__len__()
        # return 1
    def __getitem__(self,batch_num):
        FMRI_in_1,FMRI_in_2, images_out_1,images_out_2 =  self.gen_shared.__getitem__(batch_num)
        images_external_in, images_external_out = self.gen_ext_im.__getitem__(batch_num)
        return [FMRI_in_1,FMRI_in_2,images_out_1,images_out_2, images_external_out], [FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,\
            FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2]

class batch_generator_encoder_2_subj(Sequence):
    def __init__(self, X_1, Y_1, X_2, Y_2, train_labels = None, batch_paired = 48, batch_unpaired = 16, max_shift_enc = 5, img_len = 112
                 , frac_test = 3,ext_dir =  '/net/mraid11/export/data/navvew/SSReconstnClass/data/ImageNet_Files/train_images', num_ext_per_class=150, ignore_test_fmri_labels = None):
        self.batch_size     = batch_paired+batch_unpaired
        self.gen_shared        = batch_generator_2_subj(X_1,X_2, Y_1,Y_2, batch_size=batch_paired, labels = train_labels)
        self.gen_ext_im     = batch_generator_external_images(img_size = img_len, batch_size=batch_paired,ext_dir = ext_dir, num_ext_per_class=num_ext_per_class)
        self.batch_paired   = batch_paired
        self.batch_unpaired = batch_unpaired
    def on_epoch_end(self):
        self.gen_shared.on_epoch_end()
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.gen_shared.__len__()
        # return 1
    def __getitem__(self,batch_num):
        FMRI_in_1,FMRI_in_2, images_out_1,images_out_2 =  self.gen_shared.__getitem__(batch_num)
        images_external_in, images_external_out = self.gen_ext_im.__getitem__(batch_num)
        return [FMRI_in_1,FMRI_in_2,images_out_1,images_out_2, images_external_out], [FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,\
            FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2]

class batch_generator_encoder_2_subj_new(Sequence):
    def __init__(self, X_1, Y_1, X_2, Y_2, train_labels = None, batch_paired = 48, batch_unpaired = 16, max_shift_enc = 5, img_len = 112
                 , frac_test = 3,ext_dir =  '/net/mraid11/export/data/navvew/SSReconstnClass/data/ImageNet_Files/train_images', num_ext_per_class=150, ignore_test_fmri_labels = None):
        self.batch_size     = batch_paired+batch_unpaired
        self.gen_shared     = batch_generator_2_subj_new(X_1,X_2, Y_1,Y_2, batch_size=batch_paired, labels = train_labels,max_shift = max_shift_enc)
        self.gen_ext_im     = batch_generator_external_images(img_size = img_len, batch_size=batch_paired,ext_dir = ext_dir, num_ext_per_class=num_ext_per_class)
        self.batch_paired   = batch_paired
        self.batch_unpaired = batch_unpaired
    def on_epoch_end(self):
        self.gen_shared.on_epoch_end()
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.gen_shared.__len__()
        # return 1
    def __getitem__(self,batch_num):
        FMRI_in_1,FMRI_in_2, images_out_1,images_out_2 =  self.gen_shared.__getitem__(batch_num)
        images_external_in, images_external_out = self.gen_ext_im.__getitem__(batch_num)
        return [FMRI_in_1,FMRI_in_2,images_out_1,images_out_2, images_external_out], [FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,\
            FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2]

class batch_generator_encoder_3_subj(Sequence):
    def __init__(self, X_1, Y_1, X_2, Y_2, X_3, Y_3, train_labels = None, batch_paired = 48, batch_unpaired = 16, max_shift_enc = 5, img_len = 112
                 , frac_test = 3,ext_dir =  '/net/mraid11/export/data/navvew/SSReconstnClass/data/ImageNet_Files/train_images', num_ext_per_class=150, ignore_test_fmri_labels = None):
        self.batch_size     = batch_paired+batch_unpaired
        self.gen_shared        = batch_generator_3_subj_shared(X_1,Y_1,X_2,Y_2,X_3, Y_3, batch_size=batch_paired, labels = train_labels)
        self.gen_ext_im     = batch_generator_external_images(img_size = img_len, batch_size=batch_paired,ext_dir = ext_dir, num_ext_per_class=num_ext_per_class)
        self.batch_paired   = batch_paired
        self.batch_unpaired = batch_unpaired
    def on_epoch_end(self):
        self.gen_shared.on_epoch_end()
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.gen_shared.__len__()
        # return 1
    def __getitem__(self,batch_num):
        FMRI_in_1,FMRI_in_2,FMRI_in_3, images_out_1,images_out_2,images_out_3 =  self.gen_shared.__getitem__(batch_num)
        images_external_in, images_external_out = self.gen_ext_im.__getitem__(batch_num)
        return [FMRI_in_1,FMRI_in_2,FMRI_in_3,images_out_1,images_out_2,images_out_3, images_external_out], [FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,\
            FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2]

class batch_generator_encoder_3_subj(Sequence):
    def __init__(self, X_1, Y_1, X_2, Y_2, X_3, Y_3, train_labels = None, batch_paired = 48, batch_unpaired = 16, max_shift_enc = 5, img_len = 112
                 , frac_test = 3,ext_dir =  '/net/mraid11/export/data/navvew/SSReconstnClass/data/ImageNet_Files/train_images', num_ext_per_class=150, ignore_test_fmri_labels = None):
        self.batch_size     = batch_paired+batch_unpaired
        self.gen_shared        = batch_generator_3_subj_shared(X_1,Y_1,X_2,Y_2,X_3, Y_3, batch_size=batch_paired, labels = train_labels)
        self.gen_ext_im     = batch_generator_external_images(img_size = img_len, batch_size=batch_paired,ext_dir = ext_dir, num_ext_per_class=num_ext_per_class)
        self.batch_paired   = batch_paired
        self.batch_unpaired = batch_unpaired
    def on_epoch_end(self):
        self.gen_shared.on_epoch_end()
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.gen_shared.__len__()
        # return 1
    def __getitem__(self,batch_num):
        FMRI_in_1,FMRI_in_2,FMRI_in_3, images_out_1,images_out_2,images_out_3 =  self.gen_shared.__getitem__(batch_num)
        images_external_in, images_external_out = self.gen_ext_im.__getitem__(batch_num)
        return [FMRI_in_1,FMRI_in_2,FMRI_in_3,images_out_1,images_out_2,images_out_3, images_external_out], [FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,\
            FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2]
class batch_generator_encoder_2_subj_NSD(Sequence):
    def __init__(self, X_1, Y_1, X_2, Y_2,X_1_s, Y_1_s, X_2_s, Y_2_s, train_labels = None, batch_paired = 48, batch_unpaired = 16, 
                 max_shift_enc = 5, img_len = 112, frac_test = 3,
                 ext_dir =  '/net/mraid11/export/data/navvew/SSReconstnClass/data/ImageNet_Files/train_images', num_ext_per_class=150, 
                 ignore_test_fmri_labels = None, non_shared=True, ext_imgs=True,to_grey = 0 ):
        self.batch_size     = batch_paired+batch_unpaired
        self.gen_shared        = batch_generator_2_subj(X_1_s,X_2_s, Y_1_s, Y_2_s, batch_size=batch_paired, labels = train_labels)
        self.gen_dec_1        = batch_generator_enc(X_1, Y_1, batch_size=batch_paired, labels = train_labels,max_shift = max_shift_enc) if non_shared else None
        self.gen_dec_2        = batch_generator_enc(X_2, Y_2, batch_size=batch_paired, labels = train_labels,max_shift = max_shift_enc) if non_shared else None
        self.gen_ext_im     = batch_generator_external_images(img_size = img_len, batch_size=batch_paired,ext_dir = ext_dir, num_ext_per_class=num_ext_per_class,to_grey=to_grey) if ext_imgs else None
        self.batch_paired   = batch_paired
        self.batch_unpaired = batch_unpaired
    def on_epoch_end(self):
        self.gen_shared.on_epoch_end()
        self.gen_dec_1.on_epoch_end()
        self.gen_dec_2.on_epoch_end()
    def __len__(self):
        'Denotes the number of batches per epoch'
        return min(self.gen_shared.__len__(), self.gen_dec_2.__len__())
        # return 1
    def __getitem__(self,batch_num):
        FMRI_in_1_s,FMRI_in_2_s, images_out_1_s,images_out_2_s =  self.gen_shared.__getitem__(batch_num)
        images_out_1,FMRI_in_1 =  self.gen_dec_1.__getitem__(batch_num) if self.gen_dec_1 is not None else [tf.zeros_like(images_out_1_s, dtype=images_out_1_s.dtype), tf.zeros_like(FMRI_in_1_s, dtype=FMRI_in_1_s.dtype)]
        images_out_2,FMRI_in_2 =  self.gen_dec_2.__getitem__(batch_num) if self.gen_dec_2 is not None else [tf.zeros_like(images_out_1_s, dtype=images_out_1_s.dtype), tf.zeros_like(FMRI_in_1_s, dtype=FMRI_in_1_s.dtype)]
        images_external_in, images_external_out = self.gen_ext_im.__getitem__(batch_num) if self.gen_ext_im is not None else [tf.zeros_like(images_out_1_s, dtype=images_out_1_s.dtype)]*2

        return [FMRI_in_1,FMRI_in_2,images_out_1,images_out_2,FMRI_in_1_s,FMRI_in_2_s, images_out_1_s,images_out_2_s, images_external_out], [FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2]

class batch_generator_encoder_2_subj_NSD_new(Sequence):
    def __init__(self, X_1, Y_1, X_2, Y_2,X_1_s, Y_1_s, X_2_s, Y_2_s,Y_1_avg,Y_1_avg_s,Y_2_avg,Y_2_avg_s, train_labels = None, batch_paired = 48, batch_unpaired = 16, max_shift_enc = 5, img_len = 112
                 , frac_test = 3,ext_dir =  '/net/mraid11/export/data/navvew/SSReconstnClass/data/ImageNet_Files/train_images', num_ext_per_class=150, ignore_test_fmri_labels = None):
        self.batch_size     = batch_paired+batch_unpaired
        self.gen_shared        = batch_generator_2_subj_with_add_avg(X_1_s,X_2_s, Y_1_s, Y_2_s,Y_1_avg_s,Y_2_avg_s, batch_size=batch_paired, labels = train_labels)
        self.gen_dec_1        = batch_generator_dec_with_avg(X_1, Y_1,Y_1_avg, batch_size=batch_paired, labels = train_labels)
        self.gen_dec_2        = batch_generator_dec_with_avg(X_2, Y_2,Y_2_avg, batch_size=batch_paired, labels = train_labels)
        self.gen_ext_im     = batch_generator_external_images(img_size = img_len, batch_size=batch_paired,ext_dir = ext_dir, num_ext_per_class=num_ext_per_class)
        self.batch_paired   = batch_paired
        self.batch_unpaired = batch_unpaired
        self.shared_len = self.gen_shared.__len__()
    def on_epoch_end(self):
        self.gen_shared.on_epoch_end()
        self.gen_dec_1.on_epoch_end()
        self.gen_dec_2.on_epoch_end()
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.gen_shared.__len__()
        # return 1
    def __getitem__(self,batch_num):
        FMRI_in_1_s,FMRI_in_2_s, images_out_1_s,images_out_2_s,FMRI_in_1_avg_s,FMRI_in_2_avg_s =  self.gen_shared.__getitem__(batch_num)
        FMRI_in_1,images_out_1,FMRI_in_1_avg =  self.gen_dec_1.__getitem__(batch_num)
        FMRI_in_2,images_out_2,FMRI_in_2_avg =  self.gen_dec_2.__getitem__(batch_num)
        images_external_in, images_external_out = self.gen_ext_im.__getitem__(batch_num)

        return [FMRI_in_1,FMRI_in_2,images_out_1,images_out_2,FMRI_in_1_s,FMRI_in_2_s, images_out_1_s,images_out_2_s, images_external_out,FMRI_in_1_avg,FMRI_in_2_avg,\
            FMRI_in_1_avg_s,FMRI_in_2_avg_s], [FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,\
            FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2]

class batch_generator_subj_transf_NSD(Sequence):
    def __init__(self, X_1, Y_1, X_2, Y_2,X_1_s, Y_1_s, X_2_s, Y_2_s, train_labels = None, batch_paired = 48, batch_unpaired = 16, max_shift_enc = 5, img_len = 112
                 , frac_test = 3,ext_dir =  '/net/mraid11/export/data/navvew/SSReconstnClass/data/ImageNet_Files/train_images', num_ext_per_class=150, ignore_test_fmri_labels = None):
        self.batch_size     = batch_paired+batch_unpaired
        self.gen_shared        = batch_generator_2_subj(X_1_s,X_2_s, Y_1_s, Y_2_s, batch_size=batch_paired, labels = train_labels)
        self.gen_dec_1        = batch_generator_dec(X_1, Y_1, batch_size=batch_paired, labels = train_labels)
        self.gen_dec_2        = batch_generator_dec(X_2, Y_2, batch_size=batch_paired, labels = train_labels)
        self.gen_ext_im     = batch_generator_external_images(img_size = img_len, batch_size=batch_paired,ext_dir = ext_dir, num_ext_per_class=num_ext_per_class)
        self.batch_paired   = batch_paired
        self.batch_unpaired = batch_unpaired
    def on_epoch_end(self):
        self.gen_shared.on_epoch_end()
        self.gen_dec_1.on_epoch_end()
        self.gen_dec_2.on_epoch_end()
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.gen_shared.__len__()
        # return 1
    def __getitem__(self,batch_num):
        FMRI_in_1_s,FMRI_in_2_s, images_out_1_s,images_out_2_s =  self.gen_shared.__getitem__(batch_num)
        FMRI_in_1,images_out_1 =  self.gen_dec_1.__getitem__(batch_num)
        FMRI_in_2,images_out_2 =  self.gen_dec_2.__getitem__(batch_num)
        images_external_in, images_external_out = self.gen_ext_im.__getitem__(batch_num)

        return [FMRI_in_1,FMRI_in_2,images_out_1,images_out_2,FMRI_in_1_s,FMRI_in_2_s, images_out_1_s,images_out_2_s, images_external_out], [FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,\
            FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2]


class batch_generator_subj_transf_NSD_new(Sequence):
    def __init__(self, X_1, Y_1, X_2, Y_2,X_1_s, Y_1_s, X_2_s, Y_2_s,Y_1_avg,Y_1_avg_s,Y_2_avg,Y_2_avg_s,\
         train_labels = None, batch_paired = 48, batch_unpaired = 16, max_shift_enc = 5, img_len = 112
                 , frac_test = 3,ext_dir =  '/net/mraid11/export/data/navvew/SSReconstnClass/data/ImageNet_Files/train_images', num_ext_per_class=150, ignore_test_fmri_labels = None):
        self.batch_size     = batch_paired+batch_unpaired
        self.gen_shared        = batch_generator_2_subj_with_add_avg(X_1_s,X_2_s, Y_1_s, Y_2_s,Y_1_avg_s,Y_2_avg_s, batch_size=batch_paired, labels = train_labels)
        self.gen_dec_1        = batch_generator_dec_with_avg(X_1, Y_1,Y_1_avg, batch_size=batch_paired, labels = train_labels)
        self.gen_dec_2        = batch_generator_dec_with_avg(X_2, Y_2,Y_2_avg, batch_size=batch_paired, labels = train_labels)
        self.gen_ext_im     = batch_generator_external_images(img_size = img_len, batch_size=batch_paired,ext_dir = ext_dir, num_ext_per_class=num_ext_per_class)
        self.batch_paired   = batch_paired
        self.batch_unpaired = batch_unpaired
    def on_epoch_end(self):
        self.gen_shared.on_epoch_end()
        self.gen_dec_1.on_epoch_end()
        self.gen_dec_2.on_epoch_end()
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.gen_shared.__len__()
        # return 1
    def __getitem__(self,batch_num):
        FMRI_in_1_s,FMRI_in_2_s, images_out_1_s,images_out_2_s,FMRI_in_1_avg_s,FMRI_in_2_avg_s =  self.gen_shared.__getitem__(batch_num)
        FMRI_in_1,images_out_1,FMRI_in_1_avg =  self.gen_dec_1.__getitem__(batch_num)
        FMRI_in_2,images_out_2,FMRI_in_2_avg =  self.gen_dec_2.__getitem__(batch_num)
        images_external_in, images_external_out = self.gen_ext_im.__getitem__(batch_num)

        return [FMRI_in_1,FMRI_in_2,images_out_1,images_out_2,FMRI_in_1_s,FMRI_in_2_s, images_out_1_s,images_out_2_s, images_external_out,FMRI_in_1_avg,FMRI_in_2_avg,\
            FMRI_in_1_avg_s,FMRI_in_2_avg_s],\
             [FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,\
            FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2,FMRI_in_2]





class batch_generator_encdec_with_augm(Sequence):
    def __init__(self, X, Y, Y_test, test_labels, train_labels = None, batch_paired = 48, batch_unpaired = 16, max_shift_enc = 5, img_len = 112
                 , frac_test = 3,ext_dir =  '/net/mraid11/export/data/navvew/SSReconstnClass/data/ImageNet_Files/train_images', num_ext_per_class=150, ignore_test_fmri_labels = None):
        self.batch_size     = batch_paired+batch_unpaired
        self.gen_dec        = batch_generator_dec(X, Y, batch_size=batch_paired, labels = train_labels)
        self.gen_enc        = batch_generator_enc(X, Y, batch_size=batch_paired,max_shift = max_shift_enc, labels = train_labels)
        self.gen_ext_im     = batch_generator_external_images(img_size = img_len, batch_size=batch_unpaired,ext_dir = ext_dir, num_ext_per_class=num_ext_per_class)
        self.gen_test_fmri  = batch_generator_test_fmri(Y_test,test_labels, batch_size=batch_unpaired, frac =frac_test, ignore_labels = ignore_test_fmri_labels)
        self.batch_paired   = batch_paired
        self.batch_unpaired = batch_unpaired

    def on_epoch_end(self):
        self.gen_dec.on_epoch_end()
        self.gen_enc.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.gen_dec.__len__()
        # return 1
    def __getitem__(self,batch_num):
        y_in, x_out =  self.gen_dec.__getitem__(batch_num)
        x_in, y_out = self.gen_enc.__getitem__(batch_num)
        x_ext_in, x_ext_out = self.gen_ext_im.__getitem__(batch_num)
        y_test_avg = self.gen_test_fmri.__getitem__(batch_num)

        x_in = np.concatenate([x_in, x_ext_in], axis=0)
        x_out = np.concatenate([x_out,x_ext_out], axis=0)
        y_in = np.concatenate([y_in, y_test_avg], axis=0)
        y_out = np.concatenate([y_out, y_test_avg], axis=0)
        mode = np.concatenate([np.ones([self.batch_paired, 1]), np.zeros([self.batch_unpaired, 1])], axis=0)
        return [y_in, x_in, mode], [x_out, y_out,x_out]


