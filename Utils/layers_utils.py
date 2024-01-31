import numpy as np
def calc_corr(FMRI_1,FMRI_2):
    FMRI_1 = FMRI_1 - FMRI_1.mean(0)
    FMRI_1 = FMRI_1 / (np.linalg.norm(FMRI_1,2,0)+ 0.0001)
    FMRI_2 = FMRI_2 - FMRI_2.mean(0)
    FMRI_2 = FMRI_2 / (np.linalg.norm(FMRI_2,2,0)+ 0.0001)
    corr = np.transpose(FMRI_1) @ FMRI_2
    return corr

def get_mask_corr(corr_mat,neighbors = 8):
    sort_ind = np.argsort(np.abs(corr_mat),1)
    mask = np.zeros(sort_ind.shape)
    for i in range(mask.shape[0]):
        mask[i,sort_ind[i,-neighbors:]] = 1
    return mask

def get_subjects_corr_map(FMRI_1,FMRI_2,num_n=3): 
    dist = np.abs(np.transpose(calc_corr(FMRI_1,FMRI_2))) # here put the corr matrix  ## CHange only here
    conn = np.argsort(dist, axis=1)[:, -num_n:] # choose num_n neighbors
    map = np.zeros([FMRI_2.shape[1] * num_n]) # mapping without full mamtrix
    for i in range(conn.shape[0]):
        for j in range(conn.shape[1]):
            map[i * num_n + j] = conn[i, j]
    return map
