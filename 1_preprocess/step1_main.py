import os
import numpy as np
from scipy.io import loadmat
from scipy.ndimage.interpolation import zoom
from skimage import measure
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
from preprep import step1_python
import warnings
import nibabel as nib
import argparse
import pandas as pd

def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>2*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3,1)  
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10) 
    return dilatedMask


def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg

def resample(imgs, spacing, new_spacing,order = 2):
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
        return imgs, true_spacing
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')

def savenpy(name,prep_folder,data_path,use_existing=True):  
    '''
    name: the file name (#name#.nii.gz)
    prep_folder: the folder to store preprocess result
    data_path: file path
    '''
    resolution = np.array([1,1,1])
    
    if use_existing:
        if  os.path.exists(os.path.join(prep_folder,name+'_clean.npy')):
            print(name+' had been done')
            return
    try:
        im, m1, m2, spacing = step1_python(data_path)
        Mask = m1+m2
        
        newshape = np.round(np.array(Mask.shape)*spacing/resolution)
        xx,yy,zz= np.where(Mask)
        box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
        box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
        box = np.floor(box).astype('int')
        margin = 5
        extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
        extendbox = extendbox.astype('int')

        convex_mask = m1
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1+dm2
        Mask = m1+m2
        extramask = dilatedMask ^ Mask
        bone_thresh = 210
        pad_value = 170

        im[np.isnan(im)]=-2000
        sliceim = lumTrans(im)
        sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
        bones = sliceim*extramask>bone_thresh
        sliceim[bones] = pad_value
        sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
        sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                    extendbox[1,0]:extendbox[1,1],
                    extendbox[2,0]:extendbox[2,1]]
        sliceim = sliceim2[np.newaxis,...]
        #save as npy
        np.save(os.path.join(prep_folder,name+'_clean'),sliceim)

        #Save as nifti
        sliceim=sliceim.reshape(sliceim.shape[-3], sliceim.shape[-2], sliceim.shape[-1])
        matr=np.array([[0,0,-1,0],[0,-1,0,0],[1,0,0,0],[0,0,0,1]])
        #ni_img = nib.Nifti1Image(sliceim, matr)
        #activate the following line to keep conformity with old data
        ni_img = nib.Nifti1Image(sliceim, matr)
        nib.save(ni_img, os.path.join(prep_folder,name+'_clean.nii.gz'))

    except Exception as e:
        print('bug in '+name)
        print(e)
        #raise
    print(name+' done')

if __name__ == '__main__':
    
    '''
    
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--sess_csv', type=str, default='./test.csv',
                        help='sessions want to be tested')
    parser.add_argument('--prep_root', type=str, default='/nfs/masi/gaor2/tmp/justtest',
                        help='the root for save preprocessed data')
    parser.add_argument('--ori_root', type=str, default='/nfs/masi/gaor2/tmp/justtest',
                        help='the root of original data')
    args = parser.parse_args()
    
    sess_splits = pd.read_csv(args.sess_csv)['id'].tolist()
    
    for i in range(len(sess_splits)):
        sess_id = sess_splits[i]
        savenpy(name = sess_id, prep_folder = args.prep_root,
            data_path = args.ori_root + '/' + sess_id + '.nii.gz')
        np.save(args.prep_root + '/' + sess_id + '_label.npy', np.zeros((1, 4)))
