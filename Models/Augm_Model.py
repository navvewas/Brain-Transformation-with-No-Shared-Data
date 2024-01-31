
from Models.encoder_model import *
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D,Dropout, Cropping2D,Subtract,Conv3D,Activation,Reshape,AveragePooling2D,UpSampling2D,Concatenate, Conv2DTranspose
import numpy as np
from Utils.image_loss import *
from keras.initializers import VarianceScaling
from Models.layers import locally_connected_1d_new
from keras import backend as K

def check():
    print('check')
    
def subject_transf_network(NUM_VOXELS_1,NUM_VOXELS_2, name = 'subject_transf',param_l1=0, param_l2=0):
    input_shape = (NUM_VOXELS_1,)
    model = Sequential(name = name)
    model.add(InputLayer(input_shape = input_shape))
    if param_l1:
        if param_l2:
            reg = l1_l2(l1=param_l1, l2=param_l2)
        else:
            reg = l1(param_l1)
    elif param_l2:
        reg = l2(param_l2)
    else:
        reg = None
    model.add(Dense(NUM_VOXELS_2, activation= None,kernel_regularizer=reg))
    return model

def subject_transf_network_lc(NUM_VOXELS_1,NUM_VOXELS_2,MAP,num_n, name = 'subject_transf_lc',
                              lc_l1_out = 0, lc_l2_out= 0,num_ch_lc = 1,scale = 1):
    input = Input((NUM_VOXELS_1,))
    voxel_ch = Lambda(lambda x: tf.gather(x, MAP, axis=1))(input) # Create the matrix with neighbors of values - no change
    voxel_ch = Lambda(lambda x: tf.reshape(x, (-1,NUM_VOXELS_2,num_n)))(voxel_ch) # ONly reshape - no change
    if(num_ch_lc > 1):
        out = locally_connected_1d_new(l1=lc_l1_out, l2 = lc_l2_out,kernel_initializer = VarianceScaling(scale=2.0), out = num_ch_lc)(voxel_ch)
        out = Lambda(lambda x: tf.keras.activations.relu(x))(out)
        out = locally_connected_1d(l1=lc_l1_out,kernel_initializer = keras.initializers.Ones(), out = 1)(out)

    else:
        out = locally_connected_1d(l1=lc_l1_out, l2=lc_l2_out, kernel_initializer = VarianceScaling(scale=scale), out = 1)(voxel_ch)
        # out = locally_connected_1d(l1=lc_l1_out,kernel_initializer = keras.initializers.Ones(), out = 1)(voxel_ch)

    model = Model(inputs=input, outputs=out,name =name)
    return model


def encoder_2_subj_NSD(NUM_VOXELS_1,NUM_VOXELS_2,RESOLUTION,encoder_model_1,encoder_model_2,transf_net_1_2,transf_net_2_1,shraed_non_ratio = [0.95,0.05],\
     cosine_w=0.1, transf_loss_l2=False, enc_loss_l2=False):

    FMRI_1 = Input((NUM_VOXELS_1,))
    FMRI_2 = Input((NUM_VOXELS_2,))
    images_1 = Input((RESOLUTION, RESOLUTION, 3))
    images_2 = Input((RESOLUTION, RESOLUTION, 3))    
    FMRI_1_s = Input((NUM_VOXELS_1,))
    FMRI_2_s = Input((NUM_VOXELS_2,))
    images_1_s = Input((RESOLUTION, RESOLUTION, 3))
    images_2_s = Input((RESOLUTION, RESOLUTION, 3))  
    images_external = Input((RESOLUTION, RESOLUTION, 3))

 # Lambda(lambda x:K.concatenate([x,x,x],-1))(images_external) 

    # Shared part
    rec_FMRI_2_from_FMRI_1 = transf_net_1_2(FMRI_1_s)
    rec_FMRI_1_from_FMRI_2 = transf_net_2_1(FMRI_2_s)
    enc_1_s = encoder_model_1(images_1_s)
    enc_2_s = encoder_model_2(images_2_s)
    rec_FMRI_1_from_enc_2_s = transf_net_2_1(enc_2_s)
    rec_FMRI_2_from_enc_1_s = transf_net_1_2(enc_1_s)
    # Non Shared part
    enc_1 = encoder_model_1(images_1)
    enc_2 = encoder_model_2(images_2)
    enc_1_im_2 = encoder_model_1(images_2)
    enc_2_im_1 = encoder_model_2(images_1)
    rec_FMRI_1_from_enc_2_im_1_s = transf_net_2_1(enc_2_im_1)
    rec_FMRI_2_from_enc_1_im_2_s = transf_net_1_2(enc_1_im_2)
    #External part
    im_ext_enc_1 = encoder_model_1(images_external)
    im_ext_enc_2 = encoder_model_2(images_external)
    rec_enc_1_from_enc_2_ext = transf_net_2_1(im_ext_enc_2)
    rec_enc_2_from_enc_1_ext = transf_net_1_2(im_ext_enc_1)  
    
    if transf_loss_l2:
        transf_loss = l2_loss
    else:
        transf_loss = combined_voxel_loss_noSNR #lambda fmri1_2: combined_voxel_loss_noSNR(fmri1_2[0],fmri1_2[1], cosine_w=cosine_w)
        
    if enc_loss_l2:
        enc_loss = l2_loss
    else:
        enc_loss = combined_voxel_loss_noSNR#lambda fmri1_2: combined_voxel_loss_noSNR(fmri1_2[0],fmri1_2[1], cosine_w=cosine_w)

    ## Encoding ###
    encoding_1_loss = Lambda(lambda x:enc_loss(x[0],x[1]))([enc_1,FMRI_1]) 
    encoding_1_s_loss = Lambda(lambda x:enc_loss(x[0],x[1]))([enc_1_s,FMRI_1_s]) 
    encoding_1_out = Lambda(lambda x: (shraed_non_ratio[0] * x[0] + shraed_non_ratio[1] *x[1]), name='encoding_1')([encoding_1_loss,encoding_1_s_loss])
    encoding_2_loss = Lambda(lambda x:enc_loss(x[0],x[1]))([enc_2,FMRI_2]) 
    encoding_2_s_loss = Lambda(lambda x:enc_loss(x[0],x[1]))([enc_2_s,FMRI_2_s]) 
    encoding_2_out = Lambda(lambda x: (shraed_non_ratio[0] * x[0] + shraed_non_ratio[1] *x[1]), name='encoding_2')([encoding_2_loss,encoding_2_s_loss])
    ### FMRI transformtion ###
    rec_FMRI_2_from_FMRI_1_loss = Lambda(lambda x:transf_loss(x[0],x[1]))([rec_FMRI_2_from_FMRI_1,FMRI_2_s]) 
    rec_FMRI_2_from_FMRI_1_loss_out = Lambda(lambda x: x, name='rec_FMRI_2_from_FMRI_1')(rec_FMRI_2_from_FMRI_1_loss)
    rec_FMRI_1_from_FMRI_2_loss = Lambda(lambda x:transf_loss(x[0],x[1]))([rec_FMRI_1_from_FMRI_2,FMRI_1_s]) 
    rec_FMRI_1_from_FMRI_2_loss_out = Lambda(lambda x: x, name='rec_FMRI_1_from_FMRI_2')(rec_FMRI_1_from_FMRI_2_loss)
    #enc
    rec_FMRI_1_from_enc_2_loss = Lambda(lambda x:transf_loss(x[0],x[1]))([rec_FMRI_1_from_enc_2_s,FMRI_1_s]) 
    rec_FMRI_1_from_enc_2_im_1_loss = Lambda(lambda x:transf_loss(x[0],x[1]))([rec_FMRI_1_from_enc_2_im_1_s,FMRI_1]) 
    rec_FMRI_1_from_enc_2_out = Lambda(lambda x: (x[0] + x[1])/2, name='rec_FMRI_1_from_enc_2')(\
        [rec_FMRI_1_from_enc_2_loss,rec_FMRI_1_from_enc_2_im_1_loss])  
    rec_FMRI_2_from_enc_1_loss = Lambda(lambda x:transf_loss(x[0],x[1]))([rec_FMRI_2_from_enc_1_s,FMRI_2_s]) 
    rec_FMRI_2_from_enc_1_im_2_loss = Lambda(lambda x:transf_loss(x[0],x[1]))([rec_FMRI_2_from_enc_1_im_2_s,FMRI_2]) 
    rec_FMRI_2_from_enc_1_out = Lambda(lambda x: (x[0] + x[1])/2, name='rec_FMRI_2_from_enc_1')(\
        [rec_FMRI_2_from_enc_1_loss,rec_FMRI_2_from_enc_1_im_2_loss])  
    # ext
    rec_enc_1_from_enc_2_ext_loss = Lambda(lambda x:enc_loss(x[0],x[1]))([rec_enc_1_from_enc_2_ext,im_ext_enc_1]) 
    rec_enc_1_from_enc_2_ext_out = Lambda(lambda x: x, name='rec_enc_1_from_enc_2_ext')(rec_enc_1_from_enc_2_ext_loss)  
    rec_enc_2_from_enc_1_ext_loss = Lambda(lambda x:enc_loss(x[0],x[1]))([rec_enc_2_from_enc_1_ext,im_ext_enc_2]) 
    rec_enc_2_from_enc_1_ext_out = Lambda(lambda x: x, name='rec_enc_2_from_enc_1_ext')(rec_enc_2_from_enc_1_ext_loss)  
    #cycle
    cycle_from_1 = transf_net_2_1(transf_net_1_2(FMRI_1))
    cycle_from_1_s = transf_net_2_1(transf_net_1_2(FMRI_1_s))
    cycle_from_2 = transf_net_1_2(transf_net_2_1(FMRI_2))
    cycle_from_2_s = transf_net_1_2(transf_net_2_1(FMRI_2_s))
    cycle_from_ext_1 = transf_net_2_1(transf_net_1_2(im_ext_enc_1))
    cycle_from_ext_2 = transf_net_1_2(transf_net_2_1(im_ext_enc_2))
    
    cycle_from_1_loss = Lambda(lambda x:transf_loss(x[0],x[1]))([cycle_from_1,FMRI_1]) 
    cycle_from_1_s_loss = Lambda(lambda x:transf_loss(x[0],x[1]))([cycle_from_1_s,FMRI_1_s]) 
    cycle_from_1_out = Lambda(lambda x: (x[0] + x[1])/2, name='cycle_from_1')([cycle_from_1_loss,cycle_from_1_s_loss])  
    cycle_from_2_loss = Lambda(lambda x:transf_loss(x[0],x[1]))([cycle_from_2,FMRI_2]) 
    cycle_from_2_s_loss = Lambda(lambda x:transf_loss(x[0],x[1]))([cycle_from_2_s,FMRI_2_s]) 
    cycle_from_2_out = Lambda(lambda x: (x[0] + x[1])/2, name='cycle_from_2')([cycle_from_2_loss,cycle_from_2_s_loss])  
    cycle_from_ext_1_loss = Lambda(lambda x:transf_loss(x[0],x[1]))([cycle_from_ext_1,im_ext_enc_1]) 
    cycle_from_ext_1_out = Lambda(lambda x: x, name='cycle_from_ext_1')(cycle_from_ext_1_loss)  
    cycle_from_ext_2_loss = Lambda(lambda x:transf_loss(x[0],x[1]))([cycle_from_ext_2,im_ext_enc_2]) 
    cycle_from_ext_2_out = Lambda(lambda x: x, name='cycle_from_ext_2')(cycle_from_ext_2_loss)  

    return Model(inputs=[FMRI_1,FMRI_2,images_1,images_2,FMRI_1_s,FMRI_2_s,images_1_s,images_2_s,images_external],\
        outputs=[encoding_1_out,encoding_2_out,rec_FMRI_2_from_FMRI_1_loss_out,\
        rec_FMRI_1_from_FMRI_2_loss_out,rec_FMRI_1_from_enc_2_out,rec_FMRI_2_from_enc_1_out,rec_enc_1_from_enc_2_ext_out,\
            rec_enc_2_from_enc_1_ext_out,cycle_from_1_out,cycle_from_2_out,cycle_from_ext_1_out,cycle_from_ext_2_out])
    # def on_epoch_begin(self, epoch, logs=None):
    #     print(f"Epoch {epoch + 1}/{self.params['epochs']}")
            

def nonloss(inp, **kwargs):
    return tf.zeros((1,), dtype=inp.dtype) if inp is not None else None
    

def l2_loss(fmri1, fmri2):
    return tf.keras.metrics.mean_squared_error(fmri1, fmri2)
    
    
def encoder_2_subj_NSD_only_transform(NUM_VOXELS_1,NUM_VOXELS_2,RESOLUTION,transf_net_1_2,transf_net_2_1,shraed_non_ratio = [0.95,0.05],
                                     cosine_w=0.1, transf_loss_l2=False):
    
    
    FMRI_1 = Input((NUM_VOXELS_1,))
    FMRI_2 = Input((NUM_VOXELS_2,))
    images_1 = Input((RESOLUTION, RESOLUTION, 3))
    images_2 = Input((RESOLUTION, RESOLUTION, 3))    
    FMRI_1_s = Input((NUM_VOXELS_1,))
    FMRI_2_s = Input((NUM_VOXELS_2,))
    images_1_s = Input((RESOLUTION, RESOLUTION, 3))
    images_2_s = Input((RESOLUTION, RESOLUTION, 3))  
    images_external = Input((RESOLUTION, RESOLUTION, 3))
    if transf_loss_l2:
        transf_loss = l2_loss
    else:
        transf_loss = lambda fmri1_2: combined_voxel_loss_noSNR(fmri1_2[0],fmri1_2[1], cosine_w=cosine_w)

    ## Encoding ###
    encoding_1_loss = Lambda(lambda x:nonloss(x))(FMRI_1)
    encoding_1_s_loss = Lambda(lambda x:nonloss(x))(FMRI_1)
    encoding_1_out = Lambda(lambda x:nonloss(x), name="encoding_1")(FMRI_1)
    encoding_2_loss = Lambda(lambda x:nonloss(x))(FMRI_1)
    encoding_2_s_loss = Lambda(lambda x:nonloss(x))(FMRI_1)
    encoding_2_out = Lambda(lambda x:nonloss(x), name="encoding_2")(FMRI_1)

#     ### FMRI transformtion ###

    rec_FMRI_2_from_FMRI_1 = transf_net_1_2(FMRI_1_s)
    rec_FMRI_1_from_FMRI_2 = transf_net_2_1(FMRI_2_s)
    rec_FMRI_2_from_FMRI_1_loss = Lambda(lambda x:transf_loss(x[0],x[1]))([rec_FMRI_2_from_FMRI_1,FMRI_2_s]) 
    rec_FMRI_2_from_FMRI_1_loss_out = Lambda(lambda x: x, name='rec_FMRI_2_from_FMRI_1')(rec_FMRI_2_from_FMRI_1_loss)
    rec_FMRI_1_from_FMRI_2_loss = Lambda(lambda x:transf_loss(x[0],x[1]))([rec_FMRI_1_from_FMRI_2,FMRI_1_s]) 
    rec_FMRI_1_from_FMRI_2_loss_out = Lambda(lambda x: x, name='rec_FMRI_1_from_FMRI_2')(rec_FMRI_1_from_FMRI_2_loss)
    
    #enc
    rec_FMRI_1_from_enc_2_loss = Lambda(lambda x:nonloss(x))(FMRI_1) 
    rec_FMRI_1_from_enc_2_im_1_loss = Lambda(lambda x:nonloss(x))(FMRI_1)
    rec_FMRI_1_from_enc_2_out = Lambda(lambda x:nonloss(x), name='rec_FMRI_1_from_enc_2')(FMRI_1)
    rec_FMRI_2_from_enc_1_loss = Lambda(lambda x:nonloss(x))(FMRI_1) 
    rec_FMRI_2_from_enc_1_im_2_loss = Lambda(lambda x:nonloss(x))(FMRI_1)
    rec_FMRI_2_from_enc_1_out = Lambda(lambda x: nonloss(x), name='rec_FMRI_2_from_enc_1')(FMRI_1)  
    
    # ext
    rec_enc_1_from_enc_2_ext_loss = Lambda(lambda x:nonloss(x))(FMRI_1)
    rec_enc_1_from_enc_2_ext_out = Lambda(lambda x:nonloss(x), name="rec_enc_1_from_enc_2_ext")(FMRI_1)
    rec_enc_2_from_enc_1_ext_loss = Lambda(lambda x:nonloss(x))(FMRI_1) 
    rec_enc_2_from_enc_1_ext_out = Lambda(lambda x:nonloss(x), name="rec_enc_2_from_enc_1_ext")(FMRI_1)
    
    #cycle
    cycle_from_1_s = transf_net_2_1(transf_net_1_2(FMRI_1_s)) ###############################
    cycle_from_2_s = transf_net_1_2(transf_net_2_1(FMRI_2_s)) ###############################
    
    # cycle loss
    cycle_from_1_loss = Lambda(lambda x:nonloss(x))(FMRI_1) 
    cycle_from_1_s_loss = Lambda(lambda x:transf_loss(x[0],x[1]))([cycle_from_1_s,FMRI_1_s])  ###############################
    cycle_from_1_out = Lambda(lambda x: x, name='cycle_from_1')(cycle_from_1_s_loss)  ###############################
    
    cycle_from_2_loss = Lambda(lambda x:nonloss(x))(FMRI_1)     
    cycle_from_2_s_loss = Lambda(lambda x:transf_loss(x[0],x[1]))([cycle_from_2_s,FMRI_2_s]) ###############################
    cycle_from_2_out = Lambda(lambda x: x, name='cycle_from_2')(cycle_from_2_s_loss) ###############
    
    # external
    cycle_from_ext_1_loss = Lambda(lambda x:nonloss(x))(FMRI_1) 
    cycle_from_ext_1_out = Lambda(lambda x: nonloss(x), name='cycle_from_ext_1')(FMRI_1)      
    cycle_from_ext_2_loss = Lambda(lambda x: nonloss(x))(FMRI_1) 
    cycle_from_ext_2_out = Lambda(lambda x: nonloss(x), name='cycle_from_ext_2')(FMRI_1)

    return Model(inputs=[FMRI_1,FMRI_2,images_1,images_2,FMRI_1_s,FMRI_2_s,images_1_s,images_2_s,images_external],\
        outputs=[encoding_1_out,encoding_2_out,rec_FMRI_2_from_FMRI_1_loss_out,\
                 rec_FMRI_1_from_FMRI_2_loss_out,rec_FMRI_1_from_enc_2_out,rec_FMRI_2_from_enc_1_out,rec_enc_1_from_enc_2_ext_out,\
            rec_enc_2_from_enc_1_ext_out,cycle_from_1_out,cycle_from_2_out,cycle_from_ext_1_out,cycle_from_ext_2_out])





def encoder_2_subj_NSD_new(NUM_VOXELS_1,NUM_VOXELS_2,RESOLUTION,encoder_model_1,encoder_model_2,transf_net_1_2,transf_net_2_1,shraed_non_ratio = [0.95,0.05],\
     cosine_w=0.1, transf_loss_l2=False, enc_loss_l2=False):

    FMRI_1 = Input((NUM_VOXELS_1,))
    FMRI_2 = Input((NUM_VOXELS_2,))
    images_1 = Input((RESOLUTION, RESOLUTION, 3))
    images_2 = Input((RESOLUTION, RESOLUTION, 3))    
    FMRI_1_s = Input((NUM_VOXELS_1,))
    FMRI_2_s = Input((NUM_VOXELS_2,))
    images_1_s = Input((RESOLUTION, RESOLUTION, 3))
    images_2_s = Input((RESOLUTION, RESOLUTION, 3))  
    images_external = Input((RESOLUTION, RESOLUTION, 3))

 # Lambda(lambda x:K.concatenate([x,x,x],-1))(images_external) 

    # Shared part
    rec_FMRI_2_from_FMRI_1 = transf_net_1_2(FMRI_1_s)
    rec_FMRI_1_from_FMRI_2 = transf_net_2_1(FMRI_2_s)
    enc_1_s = encoder_model_1(images_1_s)
    enc_2_s = encoder_model_2(images_2_s)
    rec_FMRI_1_from_enc_2_s = transf_net_2_1(enc_2_s)
    rec_FMRI_2_from_enc_1_s = transf_net_1_2(enc_1_s)
    # Non Shared part
    enc_1 = encoder_model_1(images_1)
    enc_2 = encoder_model_2(images_2)
    enc_1_im_2 = encoder_model_1(images_2)
    enc_2_im_1 = encoder_model_2(images_1)
    rec_FMRI_1_from_enc_2_im_1_s = transf_net_2_1(enc_2_im_1)
    rec_FMRI_2_from_enc_1_im_2_s = transf_net_1_2(enc_1_im_2)
    #External part
    im_ext_enc_1 = encoder_model_1(images_external)
    im_ext_enc_2 = encoder_model_2(images_external)
    rec_enc_1_from_enc_2_ext = transf_net_2_1(im_ext_enc_2)
    rec_enc_2_from_enc_1_ext = transf_net_1_2(im_ext_enc_1)  
    
    if transf_loss_l2:
        transf_loss = l2_loss
    else:
        transf_loss = combined_voxel_loss_noSNR #lambda fmri1_2: combined_voxel_loss_noSNR(fmri1_2[0],fmri1_2[1], cosine_w=cosine_w)
        
    if enc_loss_l2:
        enc_loss = l2_loss
    else:
        enc_loss = combined_voxel_loss_noSNR#lambda fmri1_2: combined_voxel_loss_noSNR(fmri1_2[0],fmri1_2[1], cosine_w=cosine_w)

    ## Encoding ###
    encoding_1_loss = Lambda(lambda x:enc_loss(x[0],x[1]))([enc_1,FMRI_1]) 
    encoding_1_s_loss = Lambda(lambda x:enc_loss(x[0],x[1]))([enc_1_s,FMRI_1_s]) 
    encoding_1_out = Lambda(lambda x: (shraed_non_ratio[0] * x[0] + shraed_non_ratio[1] *x[1]), name='encoding_1')([encoding_1_loss,encoding_1_s_loss])
    encoding_2_loss = Lambda(lambda x:enc_loss(x[0],x[1]))([enc_2,FMRI_2]) 
    encoding_2_s_loss = Lambda(lambda x:enc_loss(x[0],x[1]))([enc_2_s,FMRI_2_s]) 
    encoding_2_out = Lambda(lambda x: (shraed_non_ratio[0] * x[0] + shraed_non_ratio[1] *x[1]), name='encoding_2')([encoding_2_loss,encoding_2_s_loss])
    ### FMRI transformtion ###
    rec_FMRI_2_from_FMRI_1_loss = Lambda(lambda x:transf_loss(x[0],x[1]))([rec_FMRI_2_from_FMRI_1,FMRI_2_s]) 
    rec_FMRI_2_from_FMRI_1_loss_out = Lambda(lambda x: x, name='rec_FMRI_2_from_FMRI_1')(rec_FMRI_2_from_FMRI_1_loss)
    rec_FMRI_1_from_FMRI_2_loss = Lambda(lambda x:transf_loss(x[0],x[1]))([rec_FMRI_1_from_FMRI_2,FMRI_1_s]) 
    rec_FMRI_1_from_FMRI_2_loss_out = Lambda(lambda x: x, name='rec_FMRI_1_from_FMRI_2')(rec_FMRI_1_from_FMRI_2_loss)
    #enc
    rec_FMRI_1_from_enc_2_loss = Lambda(lambda x:transf_loss(x[0],x[1]))([rec_FMRI_1_from_enc_2_s,FMRI_1_s]) 
    rec_FMRI_1_from_enc_2_im_1_loss = Lambda(lambda x:transf_loss(x[0],x[1]))([rec_FMRI_1_from_enc_2_im_1_s,FMRI_1]) 
    rec_FMRI_1_from_enc_2_out = Lambda(lambda x: (x[0] + x[1])/2, name='rec_FMRI_1_from_enc_2')(\
        [rec_FMRI_1_from_enc_2_loss,rec_FMRI_1_from_enc_2_im_1_loss])  
    rec_FMRI_2_from_enc_1_loss = Lambda(lambda x:transf_loss(x[0],x[1]))([rec_FMRI_2_from_enc_1_s,FMRI_2_s]) 
    rec_FMRI_2_from_enc_1_im_2_loss = Lambda(lambda x:transf_loss(x[0],x[1]))([rec_FMRI_2_from_enc_1_im_2_s,FMRI_2]) 
    rec_FMRI_2_from_enc_1_out = Lambda(lambda x: (x[0] + x[1])/2, name='rec_FMRI_2_from_enc_1')(\
        [rec_FMRI_2_from_enc_1_loss,rec_FMRI_2_from_enc_1_im_2_loss])  
    # ext
    rec_enc_1_from_enc_2_ext_loss = Lambda(lambda x:enc_loss(x[0],x[1]))([rec_enc_1_from_enc_2_ext,im_ext_enc_1]) 
    rec_enc_1_from_enc_2_ext_out = Lambda(lambda x: x, name='rec_enc_1_from_enc_2_ext')(rec_enc_1_from_enc_2_ext_loss)  
    rec_enc_2_from_enc_1_ext_loss = Lambda(lambda x:enc_loss(x[0],x[1]))([rec_enc_2_from_enc_1_ext,im_ext_enc_2]) 
    rec_enc_2_from_enc_1_ext_out = Lambda(lambda x: x, name='rec_enc_2_from_enc_1_ext')(rec_enc_2_from_enc_1_ext_loss)  
    #cycle
    cycle_from_1 = transf_net_2_1(transf_net_1_2(FMRI_1))
    cycle_from_1_s = transf_net_2_1(transf_net_1_2(FMRI_1_s))
    cycle_from_2 = transf_net_1_2(transf_net_2_1(FMRI_2))
    cycle_from_2_s = transf_net_1_2(transf_net_2_1(FMRI_2_s))
    cycle_from_ext_1 = transf_net_2_1(transf_net_1_2(im_ext_enc_1))
    cycle_from_ext_2 = transf_net_1_2(transf_net_2_1(im_ext_enc_2))
    
    cycle_from_1_loss = Lambda(lambda x:transf_loss(x[0],x[1]))([cycle_from_1,FMRI_1]) 
    cycle_from_1_s_loss = Lambda(lambda x:transf_loss(x[0],x[1]))([cycle_from_1_s,FMRI_1_s]) 
    cycle_from_1_out = Lambda(lambda x: (x[0] + x[1])/2, name='cycle_from_1')([cycle_from_1_loss,cycle_from_1_s_loss])  
    cycle_from_2_loss = Lambda(lambda x:transf_loss(x[0],x[1]))([cycle_from_2,FMRI_2]) 
    cycle_from_2_s_loss = Lambda(lambda x:transf_loss(x[0],x[1]))([cycle_from_2_s,FMRI_2_s]) 
    cycle_from_2_out = Lambda(lambda x: (x[0] + x[1])/2, name='cycle_from_2')([cycle_from_2_loss,cycle_from_2_s_loss])  
    cycle_from_ext_1_loss = Lambda(lambda x:transf_loss(x[0],x[1]))([cycle_from_ext_1,im_ext_enc_1]) 
    cycle_from_ext_1_out = Lambda(lambda x: x, name='cycle_from_ext_1')(cycle_from_ext_1_loss)  
    cycle_from_ext_2_loss = Lambda(lambda x:transf_loss(x[0],x[1]))([cycle_from_ext_2,im_ext_enc_2]) 
    cycle_from_ext_2_out = Lambda(lambda x: x, name='cycle_from_ext_2')(cycle_from_ext_2_loss)  

    return Model(inputs=[FMRI_1,FMRI_2,images_1,images_2,FMRI_1_s,FMRI_2_s,images_1_s,images_2_s,images_external],\
        outputs=[encoding_1_out,encoding_2_out,rec_FMRI_2_from_FMRI_1_loss_out,\
        rec_FMRI_1_from_FMRI_2_loss_out,rec_FMRI_1_from_enc_2_out,rec_FMRI_2_from_enc_1_out,rec_enc_1_from_enc_2_ext_out,\
            rec_enc_2_from_enc_1_ext_out,cycle_from_1_out,cycle_from_2_out,cycle_from_ext_1_out,cycle_from_ext_2_out])