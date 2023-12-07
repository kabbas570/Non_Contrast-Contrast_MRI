from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

NUM_WORKERS=0
PIN_MEMORY=True


DIM_ = 128

def same_depth(img):
    temp = np.zeros([32,DIM_,DIM_])
    temp[0:img.shape[0],:,:] = img
    return temp 


def crop_center_3D(img,cropx=DIM_,cropy=DIM_):
    z,x,y = img.shape
    startx = x//2 - cropx//2
    starty = (y)//2 - cropy//2    
    return img[:,startx:startx+cropx, starty:starty+cropy]

def Cropping_3d(img_):
    
    org_dim3 = img_.shape[0]
    org_dim1 = img_.shape[1]
    org_dim2 = img_.shape[2] 
    
    if org_dim1<DIM_ and org_dim2<DIM_:
        padding1=int((DIM_-org_dim1)//2)
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,DIM_,DIM_])
        temp[:,padding1:org_dim1+padding1,padding2:org_dim2+padding2] = img_[:,:,:]
        img_ = temp
    if org_dim1>DIM_ and org_dim2>DIM_:
        img_ = crop_center_3D(img_)        
        ## two dims are different ####
    if org_dim1<DIM_ and org_dim2>=DIM_:
        padding1=int((DIM_-org_dim1)//2)
        temp=np.zeros([org_dim3,DIM_,org_dim2])
        temp[:,padding1:org_dim1+padding1,:] = img_[:,:,:]
        img_=temp
        img_ = crop_center_3D(img_)
    if org_dim1==DIM_ and org_dim2<DIM_:
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,DIM_,DIM_])
        temp[:,:,padding2:org_dim2+padding2] = img_[:,:,:]
        img_=temp
    
    if org_dim1>DIM_ and org_dim2<DIM_:
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,org_dim1,DIM_])
        temp[:,:,padding2:org_dim2+padding2] = img_[:,:,:]
        img_ = crop_center_3D(temp)   
    return img_

def Normalization_1(img):
        mean=np.mean(img)
        std=np.std(img)
        img=(img-mean)/std
        return img 
    
    
class Dataset(Dataset):
    def __init__(self, CINE_folder,GT_folder,LGE_folder):
        self.CINE_folder = CINE_folder
        self.GT_folder = GT_folder
        self.LGE_folder = LGE_folder
        
        self.images = os.listdir(GT_folder)
       
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
                
        GT_path = os.path.join(self.GT_folder, self.images[index])
        CINE_path = os.path.join(self.CINE_folder, self.images[index][:-12]+'pixel_array_data.npy')
        LGE_path = os.path.join(self.LGE_folder, self.images[index][:-12]+'reg_lge_pixel_data.npy')


        CINE = np.load(CINE_path,allow_pickle=True)[0]
        CINE = list(CINE.values())
        CINE = np.float64(np.array(CINE))
        CINE = Cropping_3d(CINE)
        CINE = Normalization_1(CINE)
        CINE = same_depth(CINE)
        
        
        GT = np.load(GT_path,allow_pickle=True)
        GT = np.expand_dims(GT, axis=0)
        GT = Cropping_3d(GT)
        
        temp = np.zeros((4,128,128))
        
        temp[0:1,:][np.where(GT==1)] = 1
        temp[1:2,:][np.where(GT==2)] = 1
        temp[2:3,:][np.where(GT==3)] = 1
        temp[3:4,:][np.where(GT==0)] = 1

        
        LGE = np.float64(np.load(LGE_path,allow_pickle=True)[0][0])
        LGE = Normalization_1(LGE)
        LGE = np.expand_dims(LGE, axis=0)
        LGE = Cropping_3d(LGE)
        
        Class = self.images[index][:-40]
        
        print(GT_path)
        print(CINE_path)
        print(LGE_path)
        
        class_label = 0
        if Class=='ABSENT':   #  [ present-->0 and absent -->1]
            class_label = 1
        
        return  CINE,temp,LGE,class_label, self.images[index][:-4]
    
def Data_Loader(CINE_folder,GT_folder,LGE_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset(CINE_folder=CINE_folder,GT_folder=GT_folder,LGE_folder=LGE_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader

batch_size = 1
CINE_folder = r'C:\My_Data\Barts_Data\data\Data_Class_MI\my_data\five_fold_data\F1\val\CINE'
GT_folder = r'C:\My_Data\Barts_Data\data\Data_Class_MI\my_data\five_fold_data\F1\val\GTS'
LGE_folder = r'C:\My_Data\Barts_Data\data\Data_Class_MI\my_data\five_fold_data\F1\val\LGE'

val = Data_Loader(CINE_folder,GT_folder,LGE_folder,batch_size)

a = iter(val)
#a1 = next(a)
for i in range(1):
    a1 =next(a)
    cine = a1[0].numpy()
    gt = a1[1].numpy()
    lge = a1[2].numpy()
    cl_label = a1[3]
    name = a1[4]
    
#     # plt.imsave(os.path.join(r'C:\My_Data\Barts_Data\data\Data_Class_MI\my_data\VIZ\gts/',name[0]+".png"),gt[0,0,:])


plt.figure()
plt.imshow(cine[0,10,:])

for i in range(4):
    plt.figure()
    plt.imshow(gt[0,i,:])

plt.figure()
plt.imshow(lge[0,0,:])

plt.figure()
plt.imshow(a1[5][0,0,:])


# # gt = np.load(r'C:\My_Data\Barts_Data\data\Data_Class_MI\my_data\GTS\ABSENT_CON-AA166_ (85)_series1003_seg_data.npy',allow_pickle=True)
# # cine = np.load(r'C:\My_Data\Barts_Data\data\Data_Class_MI\my_data\CINE\ABSENT_CON-AA166_ (85)_series1003_pixel_array_data.npy',allow_pickle=True)[0]
# # cine = list(cine.values())
# # cine = np.array(cine)
# # c= cine[10]
# # b = np.float64(np.load(r'C:\My_Data\Barts_Data\data\Data_Class_MI\my_data\LGE\ABSENT_CON-AA166_ (85)_series1003_reg_lge_pixel_data.npy',allow_pickle=True)[0][0])
# # cl_label = a1[3].numpy()
# # name = a1[4]

# import torch
# import kornia
# three =np.zeros(gt.shape)
# three[np.where(gt!=0)] = 1
# three = np.concatenate((three,)*3, axis=1)

# three =torch.tensor(three)
# magnitude, edges=kornia.filters.canny(three, low_threshold=0.1, high_threshold=0.2, kernel_size=(7, 7), sigma=(1, 1), hysteresis=True, eps=1e-06)

# edges = edges[0,0,:].numpy()

# def normalize(x):
#     return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))

# lge = normalize(lge)
# lge = np.stack((lge,)*3, axis=2)

# lge = lge[0,0,0,:]
# lge[np.where(edges[:,:]!=0)] = 1

# import matplotlib.pyplot as plt

# plt.figure()
# plt.imshow(lge)
