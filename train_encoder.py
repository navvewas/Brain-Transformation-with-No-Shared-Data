import os
from tables import test
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

if __name__ == '__main__':
    import sys
    os.environ["CUDA_VISIBLE_DEVICES"] =  sys.argv[1]
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from datetime import datetime
    from keras.callbacks import LearningRateScheduler, TensorBoard
    from Utils.image_loss import *
    from Utils.batch_generator import *
    from Utils.callbacks import *
    from Models.Augm_Model import encoder_2_subj_NSD
    # from bdpy.distcomp import DistComp
    from time import time
    from Utils.params.training_params_encoder import *
    from Utils.train_utils import *
    

    subject_1 = eval(sys.argv[2]) # subject 1 number
    subject_student_1_data = eval(sys.argv[3]) # Subject 1 data type : 0- ImageNet dataset,  1 - NSD
    teacher_subject = eval(sys.argv[4]) # 0 no SS, else SS with which subject
    teacher_subject_data = eval(sys.argv[5]) # Subject data: 0- ImageNet dataset,  1 - NSD
    ref = eval(sys.argv[6]) # 0 - using SS (Self Supervised), 1- refrence, not using SS
    exp_array_s = [eval(sys.argv[7])] # shared example number
    exp_array_ns = [eval(sys.argv[8])] # non-shared example number

    Voxel_to_choose,num_n_1,num_n_2,l2_reg_dct_1,l2_reg_dct_2,exp_array_s,exp_array_ns,exp_array_s_2,exp_array_ns_2,exp_teacher_s,exp_teacher_ns,\
            scale_1, scale_2,param_loss_weights,extra_save,exp_s,exp_ns,exp_t_s,exp_t_ns,exp_t,exp_s_2,exp_ns_2,l2_reg_1,l2_reg_2,extra_teacher,extra_teacher_1,extra_teacher_2,ablation_text,\
            encoder_weights_1,extra_subj_1,encoder_weights_2,extra_subj_2,transf_weights_1_to_2,\
    transf_weights_2_to_1  = prepare_all_params(encoder_transf_run,share_non_share_param,exp_array_s,exp_array_ns,exp_array_s_2,exp_array_ns_2,\
                                                exp_teacher_s_ns, subject_student_1_data,subject_student_2_data,teacher_subject_data,ref,subject_1,subject_1,teacher_subject)

    if(subject_student_1_data==1):
        dataset = 'NSD'
    else:
        dataset = 'GOD'
    print('Training Encoder Subject ' +str(subject_1) +',Dataset ' + str(dataset) + ', shared_examples=' + str(exp_s) + ', nonshared_examples='+str(exp_ns) + '  ' + \
          extra_teacher.split('_exp')[0][1:])

    # Load subjects data
    print("loading data and encoders...")
    train_FMRI_1_s,train_FMRI_2_s,NUM_VOXELS_1,NUM_VOXELS_2,encoder_model_1,encoder_model_2,train_images_1, train_FMRI_1, train_images_2,\
    train_FMRI_2,train_images_1_s, train_FMRI_1_s, train_images_2_s, train_FMRI_2_s,test_images_1, test_FMRI_1, test_images_2, test_FMRI_2,\
    test_images_1_s, test_FMRI_1_s, test_images_2_s, test_FMRI_2_s = load_data_and_encoders(encoder_transf_run,subject_student_1_data,subject_student_2_data,teacher_subject_data,\
                          subject_1,subject_1,teacher_subject,get_train_avg,exp_s,exp_s_2,exp_t_s,Voxel_to_choose,scale_1,scale_2,\
                          vgg_loss,encoder_weights_1,encoder_weights_2,ref,exp_t,exp_ns,exp_ns_2)
    # Transforatmion models
    print("creating maps and transformation models...")
    transf_net_1_to_2,transf_net_2_to_1 = get_transforamtions(encoder_transf_run,train_FMRI_1_s,train_FMRI_2_s,num_n_1,num_n_2,locally_connected,subject_student_1_data,\
                                                               subject_student_2_data,teacher_subject_data,NUM_VOXELS_1,NUM_VOXELS_2,transf_l1_reg,l2_reg_1,l2_reg_2)

    print("creating model...")
    model = encoder_2_subj_NSD(NUM_VOXELS_1, NUM_VOXELS_2, RESOLUTION, encoder_model_1, encoder_model_2,
                            transf_net_1_to_2, transf_net_2_to_1, enc_loss_l2=enc_loss_l2,
                            transf_loss_l2=transf_loss_l2)

    print("compiling model...")
    model.compile(loss={'encoding_1': combined_voxel_loss_fake, 'encoding_2': combined_voxel_loss_fake, \
                        'rec_FMRI_2_from_FMRI_1': combined_voxel_loss_fake,
                        'rec_FMRI_1_from_FMRI_2': combined_voxel_loss_fake, \
                        'rec_FMRI_1_from_enc_2': combined_voxel_loss_fake,
                        'rec_FMRI_2_from_enc_1': combined_voxel_loss_fake, \
                        'rec_enc_1_from_enc_2_ext': combined_voxel_loss_fake,
                        'rec_enc_2_from_enc_1_ext': combined_voxel_loss_fake, \
                        'cycle_from_1': combined_voxel_loss_fake, 'cycle_from_2': combined_voxel_loss_fake,
                        'cycle_from_ext_1': combined_voxel_loss_fake, \
                        'cycle_from_ext_2': combined_voxel_loss_fake}, \
                loss_weights=param_loss_weights, optimizer=Adam(lr=initia_lr, amsgrad=True))

    print("creating callbacks...")
    callback_list = []
    callback = TensorBoard()
    callback.set_model(model)
    callback_list.append(callback)
    callback_list.append(EpochProgressPrinter())
    if (encoder_transf_run ==0):
        subject_2_cur = teacher_subject
    else:
        subject_2_cur = subject_2
    # Define callbacks for evaluation and lr
    # if (encoder_transf_run == 0):
    #     callback_list.append(corr_metric_callback(train_data=[train_images_1, train_FMRI_1],
    #                                             test_data=[test_images_1, test_FMRI_1],
    #                                             encoder_model=encoder_model_1, tensorboard_cb=callback,
    #                                             num_voxels=NUM_VOXELS_1,
    #                                             name='encoder_S0' + str(subject_1)))

    reduce_lr = LearningRateScheduler(step_decay)
    callback_list.append(reduce_lr)
    # Batch generator loaders
    print("creating generators...")
    loader_train = batch_generator_encoder_2_subj_NSD(train_images_1, train_FMRI_1, train_images_2,
                                                    train_FMRI_2,
                                                    train_images_1_s, train_FMRI_1_s, train_images_2_s,
                                                    train_FMRI_2_s,
                                                    batch_paired=batch_size, batch_unpaired=batch_size,
                                                    num_ext_per_class=50,
                                                    ignore_test_fmri_labels=None, max_shift_enc=5)  # ,
    #                                                               ext_imgs=ref != 1)#, non_shared=ref != 1)

    loader_test = batch_generator_encoder_2_subj_NSD(test_images_1, test_FMRI_1, test_images_2, test_FMRI_2,
                                                    test_images_1_s, test_FMRI_1_s, test_images_2_s, test_FMRI_2_s,
                                                    batch_paired=50, batch_unpaired=batch_size,
                                                    num_ext_per_class=50,
                                                    ignore_test_fmri_labels=None, max_shift_enc=0)  # ,#)
    # Train model
    print("fitting...")
    model.fit_generator(loader_train, validation_data=loader_test, epochs=epochs, verbose=0,
                        callbacks=callback_list, workers=1, use_multiprocessing=True)  # epochs
    print('Correlation Results of the Transforatmion on the Test Set:')
    test_model_and_print(encoder_model_1,test_images_1_s,test_FMRI_2_s, bacth_size=16, num_batches=30)
    #save models
    print("saving weights...")
    if(encoder_transf_run == 1):
        transf_net_1_to_2.save_weights(transf_weights_1_to_2)
        transf_net_2_to_1.save_weights(transf_weights_2_to_1)
    else:
        encoder_model_1.save_weights(encoder_weights_1)

