from PIL import Image
import numpy as np
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_imlist(dir):
    images = []

    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(fname)
    return images

def computeBER_mth(GT,Pred,mth):
    print(Pred)
    print mth
    GTimlist = get_imlist(GT)
    nim = len(GTimlist)
    nth = len(mth)
    stats = np.zeros((nth,nim,4),dtype='float')
    for i in range(0,len(GTimlist)):
        im = GTimlist[i]
        GTim = np.asarray(Image.open(os.path.join(GT,im)).convert('L'))
        posPoints = GTim>0.5
        negPoints = GTim<=0.5
        countPos = np.sum(posPoints.astype('uint8'))
        countNeg = np.sum(negPoints.astype('uint8'))
        sz = GTim.shape
        GTim = GTim>0.5
        Predim = np.asarray(Image.open(os.path.join(Pred,im)).convert('L').resize((sz[1],sz[0]),Image.NEAREST))
        Predim.setflags(write=1)
        for j in range(0,nth):
            th = mth[j]
            tp = (Predim>th) & posPoints
            tn = (Predim<=th) & negPoints
            countTP = np.sum(tp)
            countTN = np.sum(tn)
            stats[j,i,:] = [countTP,countTN,countPos,countNeg]
    all = np.zeros((nth,5))
    for j in range(0,nth):
        posAcc = np.sum(stats[j,:,0]) / np.sum(stats[j,:,2])
        negAcc = np.sum(stats[j,:,1]) / np.sum(stats[j,:,3])
        pA = 100 - 100*posAcc
        nA = 100 -100 * negAcc
        BER = 0.5 * (2-posAcc - negAcc)*100
        acc = (np.sum(stats[j,:,0]) + np.sum(stats[j,:,1]))/(np.sum(stats[j,:,02]) + np.sum(stats[j,:,3]))
        print('ACC,BER,pA,nA,th')
        all[j,:] = [acc,BER,pA,nA,mth[j]]
        #all.append([BER,pA,nA,mth[j]])
        #print(BER,posAcc,negAcc,pA,nA)
    
    #all = np.sort(all,axis=0)
    ##sort by increasing BER
    ind = np.argsort(all[:,1])
#    print all
    
    return all[ind[0],:]

def BER(pred):
    if 'ucf' in pred.lower():
        return computeUCF_BER(pred)
    if 'sbu' in pred.lower():
        return computeSBU_BER(pred)
    if 'istd' in pred.lower():
        return computeISTD_BER(pred)

def computeUCF_BER(pred,mth=255*np.arange(0.3,0.7,0.02)):
    return computeBER_mth('/nfs/bigbox/hieule/GAN/datasets/UCF/Test/TestB',pred,mth)

def computeISTD_BER(pred,mth=255*np.arange(0,0.2,0.02)):
    return computeBER_mth('/nfs/bigbox/hieule/GAN/datasets/ISTD_Dataset/test256/test256_B',pred,mth)

def computeSBU_BER(pred,mth=255*np.arange(0,0.2,0.02)):
    return computeBER_mth('/nfs/bigbox/hieule/GAN/ECCV18_sdrm/evaluation/TestB',pred,mth)
if __name__ == "__main__":

    print computeBER_mth('/nfs/bigbox/hieule/GAN/datasets/ISTD_Dataset/test256/test256_B','/nfs/bigbox/hieule/GAN/code/BDRAR/ckpt/BDRAR/istd_prediction_3000',255*np.arange(0,0.5,0.05))
    print computeBER_mth('/nfs/bigbox/hieule/GAN/datasets/ISTD_Dataset/test256/test256_B','/nfs/bigbox/hieule/GAN/ADNET_demo/ISTDtest_512',[0,1,5,10,15])
    print computeBER_mth('/nfs/bigbox/hieule/GAN/datasets/ISTD_Dataset/test256/test256_B','/nfs/bigbox/hieule/GAN/data/CVPR19/RES_man/ISTD_test_256///sdcnn_l0_ndf32_ngf64_gr0_sm0_t0_bd0_SD0_GAN0_bs16_lr0.0002_ls280_fS256_al0_concat3unet_256_basic_sd_xavier_Concat3_256_sd_best',[0,5,10,25,50])
