from time import time
from Utils.load_utils_Kami_new import get_data_newK
from Utils.NSD_load_utils import *
from Models.encoder_model import encoder_param as encoder_param_old
from Models.encoder_model_NSD import *
import scipy.stats as stat
from Utils.layers_utils import get_subjects_corr_map
from Models.Augm_Model import subject_transf_network, subject_transf_network_lc
import warnings
warnings.filterwarnings("ignore")

dir_file_old = 'data/GOD_encoders/'
dir_file_NSD = 'data/NSD_encoders/'
dir_file_transf = 'data/Transformations/'
ablation_num = 0
def combined_voxel_loss_fake(FMRI_true, FMRI_pred):
    return FMRI_pred

def step_decay(epoch):
    lrate = 5e-4
    if(epoch>20):
        lrate = lrate * 0.1
    if (epoch > 30):
        lrate = lrate * 0.01
    if (epoch > 35):
        lrate = lrate * 0.001
    if (epoch > 50):
        lrate = lrate * 0.0001
    return lrate

def calc_fac_from_encoder_transf_run(encoder_transf_run,ref):
    if(encoder_transf_run):
        if(ref == 1):
            extra_save = '_ref'
            fac = [0,1, 0,0, 0, 0]
        elif(ablation_num == 1):
            extra_save = '_abl1'
            fac = [0,1,0,0, 0.1, 0]
        elif(ablation_num == 2):
            extra_save = '_abl2'
            fac = [0,1,5,5, 0, 0]
        else: # Full loss
            extra_save = ''
            fac = [0,1,5, 5,0.1, 0.1]
    else:
        if(ref):
            fac = [1,0,0,0,0,0] 
            extra_save = '_ref'
        elif(ablation_num ==1):# Transf basic + FMRI_enc
            fac = [1,1,1,0,1,0] 
            extra_save = '_abl1'
        elif(ablation_num ==2):# Transf basic + enc_enc
            fac = [1,1,0,1,1,0] 
            extra_save = '_abl2'
        elif(ablation_num ==3):#  Transf basic + FMRI_enc + enc_enc
            fac = [1,1,1,1,1,0] 
            extra_save = '_abl3'
        elif(ablation_num ==4):#  Transf basic + cycle external
            fac = [1,1,0,0,1,1] 
            extra_save = '_abl4'
        else:
            fac = [1,1,1,1,1,1] 
            extra_save = ''
    return fac,extra_save

def calc_data_type_param(encoder_transf_run,share_non_share_param,exp_array_s,exp_array_ns,exp_array_s_2,exp_array_ns_2,exp_teacher,s1_data,s2_data,teacher_data):
    Voxel_to_choose = 5000 # used only for old

    if(s1_data == 1):
        num_n_1 = 32
        scale_1 = 0.1
        l2_reg_dct_1 = {100: 3e-2,200:3e-2, 400:1e-2, 700:1e-2, 1600:5e-3, 3200:3e-3 ,6400:1e-3,6600:1e-3,7100:1e-3}
        if((exp_array_s[0] == -2)|(exp_array_ns[0] == -2)):
            if(share_non_share_param ==0): # Only shared
                exp_array_s = [100,200,400,700]
                exp_array_ns = [0,0,0,0]
            elif(share_non_share_param ==1):# Only non shared
                exp_array_s = [0,0,0,0,0,0,0]
                exp_array_ns = [100,200,400,700,1600, 3200,6400]      
            elif(share_non_share_param ==2):
                exp_array_s = [700,700,700,700]
                exp_array_ns = [900,2500,5700,6400]   
            elif(share_non_share_param ==3): # run all options
                exp_array_s = [0,0,0,0,0,0,0,100,200,400,700,700,700,700,700]
                exp_array_ns = [100,200,400,700,1600,3200,6400,0,0,0,0,900,2500,5700,6400]
            elif(share_non_share_param ==4): # run all options
                exp_array_s = [200,200,200,200,200,200]
                exp_array_ns = [0,500,1400,3000,6200,6400]
            else:
                exp_array_s = [700]
                exp_array_ns = [6400]   
    else:
        num_n_1 = 1000
        scale_1 = 0.5
        if((exp_array_s[0] == -2)|(exp_array_ns[0] == -2)):
            exp_array_s = [300,600,900,1200,2400,3600,4800,6000]
            exp_array_ns = [0,0,0,0,0,0,0,0]
        l2_reg_dct_1 = {300: 1e-2,600:1e-2, 900:5e-3, 1200:5e-3, 2400:1e-3, 3600:1e-3, 4800:1e-3 ,6000:1e-3}
    if(encoder_transf_run == 1):
        if(s2_data ==1):
            num_n_2 = 32
            scale_2 = 0.1
            l2_reg_dct_2 = {100: 3e-2,200:3e-2, 400:1e-2, 700:1e-2, 1600:5e-3, 3200:3e-3 ,6400:1e-3,6600:1e-3,7100:1e-3}
        else:
            num_n_2 = 1000
            scale_2 = 0.5
            l2_reg_dct_2 = {300: 1e-2,600:1e-2, 900:5e-3, 1200:5e-3, 2400:1e-3, 3600:1e-3, 4800:1e-3 ,6000:1e-3}

        if(s1_data != s2_data): # Between datasets
            if((exp_array_s_2[0] == -2)|(exp_array_ns_2[0] == -2)): # Has no input 
                if(s1_data == 1):
                   exp_array_s = [700] 
                   exp_array_ns = [6400] 
                   exp_array_s_2 = [6000] 
                   exp_array_ns_2 = [0] 
                else:
                    exp_array_s_2 = [700] 
                    exp_array_ns_2 = [6400] 
                    exp_array_s = [6000] 
                    exp_array_ns = [0]    
        elif((exp_array_s_2[0] == -2)|(exp_array_ns_2[0] == -2)): 
            exp_array_s_2 = exp_array_s
            exp_array_ns_2 = exp_array_ns
    else:
        if(teacher_data ==1):
            num_n_2 = 32
            scale_2 = 0.1
            l2_reg_dct_2 = {100: 3e-2,200:3e-2, 400:1e-2, 700:1e-2, 1600:5e-3, 3200:3e-3 ,6400:1e-3,6600:1e-3,7100:1e-3}
        else:
            num_n_2 = 1000
            scale_2 = 0.5
            l2_reg_dct_2 = {300: 1e-2,600:1e-2, 900:5e-3, 1200:5e-3, 2400:1e-3, 3600:1e-3, 4800:1e-3 ,6000:1e-3}
        exp_array_s_2 = [0]
        exp_array_ns_2 = [0]

    if(exp_teacher[0] == -1):
        exp_teacher_s =[]
        exp_teacher_ns =[]
        if(teacher_data):
            for i in range(len(exp_array_s)):
                exp_teacher_s.append(700)
                exp_teacher_ns.append(6400)
        else:
            for i in range(len(exp_array_s)):
                exp_teacher_s.append(6000)
                exp_teacher_ns.append(6000)
    else:
        exp_teacher_s  = exp_teacher[0]
        exp_teacher_ns  = exp_teacher[1]

    return Voxel_to_choose,num_n_1,num_n_2,l2_reg_dct_1,l2_reg_dct_2,exp_array_s,exp_array_ns,exp_array_s_2,exp_array_ns_2,exp_teacher_s,\
        exp_teacher_ns, scale_1, scale_2


def get_extra_teacher_text(teacher_subject,teacher_subject_data,exp_teacher_s,exp_teacher_ns,encoder_transf_run,\
    subject_1,subject_2,subject_student_1_data,subject_student_2_data,ref,ablation_num):
    if(ablation_num!=0):
        ablation_text = '_abl' + str(ablation_num)
    else:
        ablation_text =''
    if(ref ==1):
        extra_teacher = ''; extra_teacher_1 = '_ref';extra_teacher_2 = '_ref'
        return extra_teacher,extra_teacher_1,extra_teacher_2,ablation_text
    if (teacher_subject != 0):
        extra_teacher = '_with_subj' + str(teacher_subject)
        if (teacher_subject_data == 1):
            if(exp_teacher_s == -1): # Take all examples
                extra_teacher = extra_teacher + '_NSD_exp_all'
            else:
                extra_teacher = extra_teacher + '_NSD' + '_exp_s_' + str(exp_teacher_s)+ '_exp_ns_' + str(exp_teacher_ns) 
        else:
            extra_teacher = extra_teacher + '_exp_' + str(exp_teacher_s)
        extra_teacher_1 = extra_teacher
        extra_teacher_2 = extra_teacher
    else:
        extra_teacher = ''; extra_teacher_1 = '_ref';extra_teacher_2 = '_ref'
    # Transforamtion learning if teacher equal to one of the students use no improved encoders
    if( (encoder_transf_run ==1) & (subject_1 == teacher_subject) & (subject_student_1_data == teacher_subject_data)):
        extra_teacher_1 = '_ref'
    if( (encoder_transf_run ==1) & (subject_2 == teacher_subject) & (subject_student_2_data == teacher_subject_data)): 
        extra_teacher_2 = '_ref'      
    return extra_teacher,extra_teacher_1,extra_teacher_2,ablation_text




def get_param_loss_weights(fac):
    fac_0 = fac[0] # transf fmri to fmri
    fac_1 = fac[1] # transf fmri to fmri
    fac_2 = fac[2] # transf fmri to external
    fac_3 = fac[3] # transf external to external
    fac_4 = fac[4] # cycle fmri
    fac_5 = fac[5] # cycle external
    param_loss_weights = [fac_0,fac_0,    fac_1,fac_1  ,fac_2,fac_2       ,fac_3,fac_3,    fac_4,fac_4,fac_5,fac_5] 
    #                      enc      FMRI_FMRI     FMRI_enc        enc_enc                 cycle    
    return param_loss_weights


def prepare_all_params(encoder_transf_run,share_non_share_param,exp_array_s,exp_array_ns,exp_array_s_2,exp_array_ns_2,exp_teacher_s_ns,\
            subject_student_1_data,subject_student_2_data,teacher_subject_data,ref,subject_1,subject_2,teacher_subject):

        Voxel_to_choose,num_n_1,num_n_2,l2_reg_dct_1,l2_reg_dct_2,exp_array_s,exp_array_ns,exp_array_s_2,exp_array_ns_2,exp_teacher_s,exp_teacher_ns,\
        scale_1, scale_2 = calc_data_type_param(\
        encoder_transf_run,share_non_share_param,exp_array_s,exp_array_ns,exp_array_s_2,exp_array_ns_2,exp_teacher_s_ns,\
            subject_student_1_data,subject_student_2_data,teacher_subject_data)

        fac, extra_save = calc_fac_from_encoder_transf_run(encoder_transf_run,ref)
        param_loss_weights = get_param_loss_weights(fac)


        starting_time = time()
        exp_s = exp_array_s[0]
        exp_ns = exp_array_ns[0]
        exp_t_s = exp_teacher_s[0]
        exp_t_ns = exp_teacher_ns[0]
        exp_t = exp_t_s + exp_t_ns
        exp_s_2 = 0;exp_ns_2=0;
        if(encoder_transf_run ==1):
            exp_s_2 = exp_array_s_2[0]
            exp_ns_2 = exp_array_ns_2[0]
            l2_reg_2 = l2_reg_dct_2[exp_s_2 + exp_ns_2]
        elif(teacher_subject_data == 1):
            l2_reg_2 = l2_reg_dct_2[exp_t_s + exp_t_ns]
        else:
            l2_reg_2 = l2_reg_dct_2[exp_t_s]
        l2_reg_1 = l2_reg_dct_1[exp_s + exp_ns]


    
        # if(encoder_transf_run ==1):
        #     print('training transf s1=' +str(subject_1) + ', s2=' +str(s2) + ', exp_s=' + str(exp_s) + ', exp_ns='+str(exp_ns))
        # else:
        #     print('training encdoer s1=' +str(subject_1) + ', s2=' +str(teacher_subject) + ', exp_s=' + str(exp_s) + ', exp_ns='+str(exp_ns))
    
        # print(f"time since running cell: {time() - starting_time}")
        
        ### Define encoder weigths to load
        extra_teacher,extra_teacher_1,extra_teacher_2,ablation_text = get_extra_teacher_text(teacher_subject,teacher_subject_data,exp_t_s\
            ,exp_t_ns,encoder_transf_run,subject_1,subject_2,subject_student_1_data,subject_student_2_data,ref,ablation_num)
        if(subject_student_1_data == 1):
            encoder_weights_1 = dir_file_NSD + 'encoder_weights_subj' + str(subject_1) + '_NSD_exp_s_' + str(exp_s) +\
                 '_exp_ns_' + str(exp_ns) + extra_teacher_1 + ablation_text +'.hdf5'
            extra_subj_1 = '_NSD'
        else:
            encoder_weights_1 = dir_file_old + 'encoder_weights_subj' + str(subject_1) + '_exp_' + str(exp_s) +\
                  extra_teacher_1 + ablation_text + '.hdf5'  
            extra_subj_1 = ''
        if(encoder_transf_run ==1):
            if(subject_student_2_data == 1):
                encoder_weights_2 = dir_file_NSD + 'encoder_weights_subj' + str(subject_2) + '_NSD_exp_s_' + str(exp_s_2) +\
                    '_exp_ns_' + str(exp_ns_2) + extra_teacher_2 + ablation_text + '.hdf5'
                extra_subj_2 = '_NSD'
            else:
                encoder_weights_2 = dir_file_old + 'encoder_weights_subj' + str(subject_2) + '_exp_' + str(exp_s_2) +\
                    extra_teacher_2 + ablation_text +  '.hdf5'  
                extra_subj_2 = ''
            transf_weights_1_to_2 = dir_file_transf + 'transf_weights_subj' + str(subject_1) + extra_subj_1 + '_to_subj' +\
                str(subject_2) + extra_subj_2 + '_exp_' + str(exp_s) +'_'+ str(exp_ns) +'_'+ str(exp_s_2) +'_'+ \
                    str(exp_ns_2) + extra_teacher + extra_save + '.hdf5'
            transf_weights_2_to_1 = dir_file_transf + 'transf_weights_subj' + str(subject_2) + extra_subj_2 + '_to_subj' +\
                str(subject_1) + extra_subj_1 + '_exp_' + str(exp_s_2) +'_'+ str(exp_ns_2) +'_'+ str(exp_s) +'_'+ \
                    str(exp_ns) + extra_teacher + extra_save + '.hdf5'
        else:
            if(teacher_subject_data == 1):
                encoder_weights_2 = dir_file_NSD + 'encoder_weights_subj' + str(teacher_subject) + '_NSD_exp_s_' + str(exp_t_s) +\
                    '_exp_ns_' + str(exp_t_ns) + '_ref.hdf5'
                extra_subj_2 = '_NSD'
            else:
                encoder_weights_2 = dir_file_old + 'encoder_weights_subj' + str(teacher_subject) + '_exp_' + str(exp_t_s) +\
                     '_ref.hdf5'  
                extra_subj_2 = ''
            transf_weights_1_to_2 = ''
            transf_weights_2_to_1 = ''


        
        return  Voxel_to_choose,num_n_1,num_n_2,l2_reg_dct_1,l2_reg_dct_2,exp_array_s,exp_array_ns,exp_array_s_2,exp_array_ns_2,exp_teacher_s,exp_teacher_ns,\
        scale_1, scale_2,param_loss_weights,extra_save,exp_s,exp_ns,exp_t_s,exp_t_ns,exp_t,exp_s_2,exp_ns_2,l2_reg_1,l2_reg_2,extra_teacher,extra_teacher_1,extra_teacher_2,ablation_text,\
        encoder_weights_1,extra_subj_1,encoder_weights_2,extra_subj_2,transf_weights_1_to_2,transf_weights_2_to_1



def load_data_and_encoders(encoder_transf_run,subject_student_1_data,subject_student_2_data,teacher_subject_data,subject_1,subject_2,teacher_subject,get_train_avg,exp_s,exp_s_2,\
                           exp_t_s,Voxel_to_choose, scale_1,scale_2,vgg_loss,encoder_weights_1,encoder_weights_2,ref,exp_t,exp_ns,exp_ns_2):                   
        
        if(subject_student_1_data == 0):
            train_images_1, val_images_1, test_images_1, train_FMRI_1, val_FMRI_1, test_FMRI_1, test_FMRI_median_1, labels_train_1,\
            labels_val_1, ext_img_test_1, SNR_1, snr_inv_1, snr_1, voxel_loc_1 = \
                get_data_newK(subject_1, get_train_avg=get_train_avg, num_sample=exp_s, Voxel_to_choose=Voxel_to_choose)
            train_images_1_s = train_images_1[:exp_s]
            test_images_1_s = test_images_1
            train_FMRI_1_s = train_FMRI_1[:exp_s]
            test_FMRI_1_s = test_FMRI_1
        else:
            FMRI_ordered_Averaged_1,\
                Images_ordered_Averaged_1,Images_labels_Averaged_1,\
                                    train_mask_s,val_mask_s,test_mask_s,train_mask,val_mask,test_mask,\
                                    shared_mas,shared_all_indexes = Load_NSD_data(subject_1)
            train_images_1,val_images_1,test_images_1,train_FMRI_1,val_FMRI_1,test_FMRI_1,train_labels_1,val_labels_1, test_labels_1 = Split_data_new(Images_ordered_Averaged_1,\
                FMRI_ordered_Averaged_1,Images_labels_Averaged_1[0],train_mask,val_mask,test_mask)
        if(encoder_transf_run == 1):
            if(subject_student_2_data == 0):
                train_images_2, val_images_2, test_images_2, train_FMRI_2, val_FMRI_2, test_FMRI_2, test_FMRI_median_2, labels_train_2, \
                labels_val_2, ext_img_test_2, SNR_2, snr_inv_2, snr_2, voxel_loc_2 = \
                    get_data_newK(subject_2, get_train_avg=get_train_avg, num_sample=exp_s_2, Voxel_to_choose=Voxel_to_choose)
                train_images_2_s, tmp_var, tmp_var, train_FMRI_2_s, tmp_var, tmp_var, tmp_var, tmp_var, \
                tmp_var, tmp_var, tmp_var, tmp_var, tmp_var, tmp_var = \
                    get_data_newK(subject_2, get_train_avg=get_train_avg, num_sample=exp_s, Voxel_to_choose=Voxel_to_choose)
                test_images_2_s = test_images_2
                test_FMRI_2_s = test_FMRI_2
            else:
                FMRI_ordered_Averaged_2,Images_ordered_Averaged_2,Images_labels_Averaged_2,\
                                    train_mask_s,val_mask_s,test_mask_s,train_mask,val_mask,test_mask,\
                                        shared_mas,shared_all_indexes = Load_NSD_data(subject_2)
                train_images_2,val_images_2,test_images_2,train_FMRI_2,val_FMRI_2,test_FMRI_2,train_labels_2,val_labels_2, test_labels_2 = Split_data_new(Images_ordered_Averaged_2,\
                    FMRI_ordered_Averaged_2,Images_labels_Averaged_2[0],train_mask,val_mask,test_mask)
        else:
            if(teacher_subject_data == 0):
                train_images_2, val_images_2, test_images_2, train_FMRI_2, val_FMRI_2, test_FMRI_2, test_FMRI_median_2, labels_train_2, \
                labels_val_2, ext_img_test_2, SNR_2, snr_inv_2, snr_2, voxel_loc_2 = \
                    get_data_newK(teacher_subject, get_train_avg=get_train_avg, num_sample=exp_t_s, Voxel_to_choose=Voxel_to_choose)
                train_images_2_s, tmp_var, tmp_var, train_FMRI_2_s, tmp_var, tmp_var, tmp_var, tmp_var, \
                tmp_var, tmp_var, tmp_var, tmp_var, tmp_var, tmp_var = \
                    get_data_newK(teacher_subject, get_train_avg=get_train_avg, num_sample=exp_s, Voxel_to_choose=Voxel_to_choose)
                test_images_2_s = test_images_2
                test_FMRI_2_s = test_FMRI_2
            else:
                FMRI_ordered_Averaged_2,Images_ordered_Averaged_2,Images_labels_Averaged_2,\
                    train_mask_s,val_mask_s,test_mask_s,train_mask,val_mask,test_mask,\
                    shared_mas,shared_all_indexes = Load_NSD_data(teacher_subject)
                train_images_2,val_images_2,test_images_2,train_FMRI_2,val_FMRI_2,test_FMRI_2,train_labels_2,val_labels_2, test_labels_2 = Split_data_new(Images_ordered_Averaged_2,\
                    FMRI_ordered_Averaged_2,Images_labels_Averaged_2[0],train_mask,val_mask,test_mask)
        NUM_VOXELS_1 = train_FMRI_1.shape[1]
        NUM_VOXELS_2 = train_FMRI_2.shape[1]

        ### load encoders
        if(subject_student_1_data == 1):
            enc_param_1 = encoder_param(NUM_VOXELS_1,c2f_l1 = scale_1 * 1e-7,c2f_gl = scale_1 * 1e-7,lc_l1 = scale_1 * 1e-7,\
                lc_l1_out = scale_1 * 1e-7,conv_l1_reg = scale_1 * 1e-5,conv_l2_reg = scale_1 * 0.001)
        else:
            enc_param_1 = encoder_param_old(NUM_VOXELS_1,c2f_l1 = scale_1 * 5e-6,c2f_gl = scale_1 * 1e-5,lc_l1 = scale_1 * 5e-6,\
                lc_l1_out = scale_1 * 5e-2,conv_l1_reg = scale_1 * 1e-5,conv_l2_reg = scale_1 * 0.001)
        if(((encoder_transf_run == 1) & (subject_student_2_data == 1)) | ((encoder_transf_run == 0) & (teacher_subject_data == 1))):
            enc_param_2 = encoder_param(NUM_VOXELS_2,c2f_l1 = scale_2 * 1e-7,c2f_gl = scale_2 * 1e-7,lc_l1 = scale_2 * 1e-7,\
                lc_l1_out = scale_2 * 1e-7,conv_l1_reg = scale_2 * 1e-5,conv_l2_reg = scale_2 * 0.001) 
        else:
            enc_param_2 = encoder_param_old(NUM_VOXELS_2,c2f_l1 = scale_2 * 5e-6,c2f_gl = scale_2 * 1e-5,lc_l1 = scale_2 * 5e-6,\
                lc_l1_out = scale_2 * 5e-2,conv_l1_reg = scale_2 * 1e-5,conv_l2_reg = scale_2 * 0.001)
    
        encoder_model_1 = encoder_ml_seperable(enc_param_1, vgg_loss, ch_mult=1, name='encoder_1')
        encoder_model_2 = encoder_ml_seperable(enc_param_2, vgg_loss, ch_mult=1, name='encoder_2')
        encoder_model_2.trainable = False
        if(encoder_transf_run == 1):
            encoder_model_1.trainable = False
        if (ref ==0):
            encoder_model_2.load_weights(encoder_weights_2)
            if(encoder_transf_run ==1):
                encoder_model_1.load_weights(encoder_weights_1)

        
        if(((encoder_transf_run == 1) & (subject_student_1_data == 1) & (subject_student_2_data == 1)) |\
            (encoder_transf_run == 0) & (subject_student_1_data == 1) & (teacher_subject_data == 1)):
            train_images_1_s,val_images_1_s,test_images_1_s,train_FMRI_1_s,val_FMRI_1_s,test_FMRI_1_s,train_labels_1_s,val_labels_1_s, test_labels_1_s\
                = Split_data_new(Images_ordered_Averaged_1,\
                FMRI_ordered_Averaged_1,Images_labels_Averaged_1[0],train_mask_s,val_mask_s,test_mask_s,Image_labels_second_subj = Images_labels_Averaged_2[0])
            train_images_2_s,val_images_2_s,test_images_2_s,train_FMRI_2_s,val_FMRI_2_s,test_FMRI_2_s,train_labels_2_s,val_labels_2_s, test_labels_2_s\
                = Split_data_new(Images_ordered_Averaged_2,\
                FMRI_ordered_Averaged_2,Images_labels_Averaged_2[0],train_mask_s,val_mask_s,test_mask_s, Image_labels_second_subj = Images_labels_Averaged_1[0])
            mask_ns_1 = 1 - np.isin(train_labels_1,train_labels_1_s)
            train_images_1 = train_images_1[mask_ns_1 == 1]
            train_FMRI_1 = train_FMRI_1[mask_ns_1 == 1]
            mask_ns_2 = 1 - np.isin(train_labels_2,train_labels_2_s)
            train_images_2 = train_images_2[mask_ns_2 == 1]
            train_FMRI_2 = train_FMRI_2[mask_ns_2 == 1]
        elif((encoder_transf_run == 1) & (subject_student_1_data == 0) & (subject_student_2_data == 1)):
            train_images_2_s = train_images_1
            test_images_2_s = test_images_1
            train_FMRI_2_s = encoder_model_2.predict(train_images_1)
            test_FMRI_2_s = encoder_model_2.predict(test_images_1)
            exp_s_2 = train_images_2_s.shape[0]
        elif((encoder_transf_run == 1) & (subject_student_1_data == 1) & (subject_student_2_data == 0)):  
            train_images_1_s = train_images_2
            test_images_1_s = test_images_2
            train_FMRI_1_s = encoder_model_1.predict(train_images_2)
            test_FMRI_1_s = encoder_model_1.predict(test_images_2)
            exp_s = train_images_1_s.shape[0]
        elif((encoder_transf_run == 0) & (subject_student_1_data == 0) & (teacher_subject_data == 1)):
            train_images_2_s = train_images_1
            test_images_2_s = test_images_1
            train_FMRI_2_s = encoder_model_2.predict(train_images_1)
            test_FMRI_2_s = encoder_model_2.predict(test_images_1)
        elif((encoder_transf_run == 0) & (subject_student_1_data == 1) & (teacher_subject_data == 0)): 
            train_images_1_s = train_images_2
            test_images_1_s = test_images_2
            train_FMRI_1_s = encoder_model_1.predict(train_images_2)
            test_FMRI_1_s = encoder_model_1.predict(test_images_2)
    
        # Take only needed exmples
        if(subject_student_1_data == 1):
            if(exp_s != 0):
                train_images_1_s = train_images_1_s[:exp_s]
                train_FMRI_1_s = train_FMRI_1_s[:exp_s]
                train_images_1 = np.concatenate((train_images_1[:exp_ns],train_images_1_s),0)
                train_FMRI_1 = np.concatenate((train_FMRI_1[:exp_ns],train_FMRI_1_s),0) 
            else:
                num_exp = np.min((exp_ns,1000))
                train_images_1_s = train_images_1[:num_exp]    
                train_FMRI_1_s = train_FMRI_1[:num_exp]
                train_images_1 = train_images_1[:exp_ns]
                train_FMRI_1 = train_FMRI_1[:exp_ns]
        if(encoder_transf_run == 1):
            if(subject_student_2_data == 1):
                if(exp_s != 0):
                    train_images_2_s = train_images_2_s[:exp_s_2]
                    train_FMRI_2_s = train_FMRI_2_s[:exp_s_2]
                    train_images_2 = np.concatenate((train_images_2[:exp_ns_2],train_images_2_s),0)
                    train_FMRI_2 = np.concatenate((train_FMRI_2[:exp_ns_2],train_FMRI_2_s),0) 
                else:
                    train_images_2_s = train_images_1_s
                    train_FMRI_2_s = encoder_model_2.predict(train_images_1_s)
        else: 
            if(teacher_subject_data == 1):
                if(subject_student_1_data == 1):
                    if(exp_s != 0):
                        train_images_2_s = train_images_2_s[:exp_s]
                        train_FMRI_2_s = train_FMRI_2_s[:exp_s]
                        train_images_2 = np.concatenate((train_images_2[:exp_t],train_images_2_s),0)
                        train_FMRI_2 = np.concatenate((train_FMRI_2[:exp_t],train_FMRI_2_s),0) 
                    else:
                        train_images_2 = np.concatenate((train_images_2[:exp_t],train_images_2_s),0)
                        train_FMRI_2 = np.concatenate((train_FMRI_2[:exp_t],train_FMRI_2_s),0) 
                        train_images_2_s = train_images_1_s
                        train_FMRI_2_s = encoder_model_2.predict(train_images_1_s)  
                else:
                    if(exp_t == 7100):
                        train_images_2 = train_images_2
                        train_FMRI_2 = train_FMRI_2
                    else:
                        train_images_2 = train_images_2[:exp_t]
                        train_FMRI_2 = train_FMRI_2[:exp_t]
        return     train_FMRI_1_s,train_FMRI_2_s,NUM_VOXELS_1,NUM_VOXELS_2,encoder_model_1,encoder_model_2,train_images_1, train_FMRI_1, train_images_2,\
                        train_FMRI_2,train_images_1_s, train_FMRI_1_s, train_images_2_s, train_FMRI_2_s,test_images_1, test_FMRI_1, test_images_2, test_FMRI_2,\
                        test_images_1_s, test_FMRI_1_s, test_images_2_s, test_FMRI_2_s
    
def get_transforamtions(encoder_transf_run,train_FMRI_1_s,train_FMRI_2_s,num_n_1,num_n_2,locally_connected,subject_student_1_data,subject_student_2_data,teacher_subject_data,NUM_VOXELS_1,\
                        NUM_VOXELS_2,transf_l1_reg,l2_reg_1,l2_reg_2):
    map_fmri_1 =  train_FMRI_1_s
    map_fmri_2 = train_FMRI_2_s
    MAP_1_to_2 = get_subjects_corr_map(map_fmri_1, map_fmri_2, num_n=num_n_1)  # LC connection map
    MAP_1_to_2 = tf.constant(MAP_1_to_2, dtype=tf.int32)
    MAP_2_to_1 = get_subjects_corr_map(map_fmri_2, map_fmri_1, num_n=num_n_2)
    MAP_2_to_1 = tf.constant(MAP_2_to_1, dtype=tf.int32)

    if(locally_connected[subject_student_1_data]):
        transf_net_1_to_2 = subject_transf_network_lc(NUM_VOXELS_1, NUM_VOXELS_2, MAP_1_to_2, num_n=num_n_1,
                                                    lc_l1_out=transf_l1_reg, lc_l2_out=l2_reg_1, name='subject_transf_lc_1_to_2')
    else:
        transf_net_1_to_2 = subject_transf_network(NUM_VOXELS_1, NUM_VOXELS_2, name='subject_transf_1_to_2',
                                                 param_l1=transf_l1_reg, param_l2=l2_reg_1)
    cur_ind_locally = ((encoder_transf_run == 1) & subject_student_2_data) | ((encoder_transf_run == 0) & teacher_subject_data)
    if(locally_connected[cur_ind_locally]):
        transf_net_2_to_1 = subject_transf_network_lc(NUM_VOXELS_2, NUM_VOXELS_1, MAP_2_to_1, num_n=num_n_2,
                                                    lc_l1_out=transf_l1_reg, lc_l2_out=l2_reg_2, name='subject_transf_lc_2_to_1')
    else:
        transf_net_2_to_1 = subject_transf_network(NUM_VOXELS_2, NUM_VOXELS_1, name='subject_transf_2_to_1',
                                                param_l1=transf_l1_reg, param_l2=l2_reg_2)
    return transf_net_1_to_2,transf_net_2_to_1


def test_model_and_print(model,x,y, bacth_size=16, num_batches=30):
    num_batches = x.shape[0]//bacth_size + 1
    y_predict = np.zeros([0,y.shape[1]])
    for i in range(num_batches):
        inputs = x[bacth_size*i:bacth_size*(i+1)]
        pred = model.predict(inputs)
        y_predict = np.concatenate((y_predict, pred), axis=0)
    corr = np.zeros([y.shape[1]])
    for i in range(y.shape[1]):
        corr[i] = stat.pearsonr(y[:, i], y_predict[:, i])[0]
    corr = np.nan_to_num(corr)
    print('test corr median = ' +str(np.mean(corr)))
    print('test corr mean = ' +str(np.median(corr)))
    print('test corr 75 = ' +str(np.percentile(corr,75)))
    print('test corr 90 = ' +str(np.percentile(corr,90)))
    




