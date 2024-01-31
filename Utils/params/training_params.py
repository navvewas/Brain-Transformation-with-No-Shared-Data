    batch_size = 32
    RESOLUTION = 112
    get_train_avg = 1 # If to average fMRI repetitions
    initia_lr =  5e-4
    transf_loss_l2 = True
    enc_loss_l2 = True
    transf_l1_reg = 0
    ablation_num = 0 ## 0 no ablation
    locally_connected = [0,1] # [old 0/1, NSD, 0/1]
    share_non_share_param = 2
    exp_teacher_s_ns = [-1,-1] # how many example train on and how many you need to train with shared and non shared

