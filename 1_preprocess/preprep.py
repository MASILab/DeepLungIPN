import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import dicom
import os
import scipy.ndimage
import nibabel as nib
from skimage import measure, morphology

def load_scan(path):
    img=nib.load(path)
    return img #a nib object

def get_pixels_hu(slices): #slices is a nib object
    #####
    image=slices.get_data()
    image=np.swapaxes(image,0,2)
    image=np.swapaxes(image,1,2)
    image=np.rot90(image,axes=(1,2))

    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    header=slices.header
    spacing=np.array([header['pixdim'][3]]+[header['pixdim'][2]]+[header['pixdim'][1]], dtype=np.float32)
    return np.array(image, dtype=np.int16),spacing

def binarize_per_slice(image, spacing, intensity_th=-600, sigma=1, area_th=30, eccen_th=0.99, bg_patch_size=10):

    bw = np.zeros(image.shape, dtype=bool)
    
    # prepare a mask, with all corner values set to nan
    image_size = image.shape[1]
    grid_axis = np.linspace(-image_size/2+0.5, image_size/2-0.5, image_size)
    x, y = np.meshgrid(grid_axis, grid_axis)
    d = (x**2+y**2)**0.5
    nan_mask = (d<image_size/2).astype(float)
    nan_mask[nan_mask == 0] = np.nan
    for i in range(image.shape[0]):
        # Check if corner pixels are identical, if so the slice  before Gaussian filtering
        if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1:
            current_bw = scipy.ndimage.filters.gaussian_filter(np.multiply(image[i].astype('float32'), nan_mask), sigma, truncate=2.0) < intensity_th
        else:
            current_bw = scipy.ndimage.filters.gaussian_filter(image[i].astype('float32'), sigma, truncate=2.0) < intensity_th
        
        # select proper components
        label = measure.label(current_bw)
        properties = measure.regionprops(label)
        valid_label = set()
        for prop in properties:
            if prop.area * spacing[1] * spacing[2] > area_th and prop.eccentricity < eccen_th:
                valid_label.add(prop.label)
        current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
        bw[i] = current_bw
        
    return bw

def all_slice_analysis(bw, spacing, cut_num=0, vol_limit=[0.68, 8.2], area_th=6e3, dist_th=62):
    # in some cases, several top layers need to be removed first
    if cut_num > 0:
        bw0 = np.copy(bw)
        bw[-cut_num:] = False
    label = measure.label(bw, connectivity=1)
    # remove components access to corners
    mid = int(label.shape[2] / 2)
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1-cut_num, 0, 0], label[-1-cut_num, 0, -1], label[-1-cut_num, -1, 0], label[-1-cut_num, -1, -1], \
                    label[0, 0, mid], label[0, -1, mid], label[-1-cut_num, 0, mid], label[-1-cut_num, -1, mid]])
    for l in bg_label:
        label[label == l] = 0
        
    # select components based on volume
    properties = measure.regionprops(label)
    for prop in properties:
        if prop.area * spacing.prod() < vol_limit[0] * 1e6 or prop.area * spacing.prod() > vol_limit[1] * 1e6:
            label[label == prop.label] = 0
            
    # prepare a distance map for further analysis
    x_axis = np.linspace(-label.shape[1]/2+0.5, label.shape[1]/2-0.5, label.shape[1]) * spacing[1]
    y_axis = np.linspace(-label.shape[2]/2+0.5, label.shape[2]/2-0.5, label.shape[2]) * spacing[2]
    x, y = np.meshgrid(x_axis, y_axis)
    d = (x**2+y**2)**0.5
    vols = measure.regionprops(label)
    valid_label = set()
    # select components based on their area and distance to center axis on all slices
    for vol in vols:
        single_vol = label == vol.label
        slice_area = np.zeros(label.shape[0])
        min_distance = np.zeros(label.shape[0])
        for i in range(label.shape[0]):
            slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3])
            min_distance[i] = np.min(single_vol[i] * d + (1 - single_vol[i]) * np.max(d))
        
        if np.average([min_distance[i] for i in range(label.shape[0]) if slice_area[i] > area_th]) < dist_th:
            valid_label.add(vol.label)
            
    bw = np.in1d(label, list(valid_label)).reshape(label.shape)
    
    # fill back the parts removed earlier
    if cut_num > 0:
        # bw1 is bw with removed slices, bw2 is a dilated version of bw, part of their intersection is returned as final mask
        bw1 = np.copy(bw)
        bw1[-cut_num:] = bw0[-cut_num:]
        bw2 = np.copy(bw)
        bw2 = scipy.ndimage.binary_dilation(bw2, iterations=cut_num)
        bw3 = bw1 & bw2
        label = measure.label(bw, connectivity=1)
        label3 = measure.label(bw3, connectivity=1)
        l_list = list(set(np.unique(label)) - {0})
        valid_l3 = set()
        for l in l_list:
            indices = np.nonzero(label==l)
            l3 = label3[indices[0][0], indices[1][0], indices[2][0]]
            if l3 > 0:
                valid_l3.add(l3)
        bw = np.in1d(label3, list(valid_l3)).reshape(label3.shape)
    
    return bw, len(valid_label)

def fill_hole(bw):
    # fill 3d holes
    label = measure.label(~bw)
    # idendify corner components
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1, 0, 0], label[-1, 0, -1], label[-1, -1, 0], label[-1, -1, -1]])
    bw = ~np.in1d(label, list(bg_label)).reshape(label.shape)
    
    return bw




def two_lung_only(bw, spacing, max_iter=22, max_ratio=4.8):    
    def extract_main(bw, cover=0.95):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            properties.sort(key=lambda x: x.area, reverse=True)
            area = [prop.area for prop in properties]
            count = 0
            sum = 0
            while sum < np.sum(area)*cover:
                sum = sum+area[count]
                count = count+1
            filter = np.zeros(current_slice.shape, dtype=bool)
            for j in range(count):
                bb = properties[j].bbox
                filter[bb[0]:bb[2], bb[1]:bb[3]] = filter[bb[0]:bb[2], bb[1]:bb[3]] | properties[j].convex_image
            bw[i] = bw[i] & filter
           
        label = measure.label(bw)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        bw = label==properties[0].label

        return bw
    
    def fill_2d_hole(bw):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            for prop in properties:
                bb = prop.bbox
                current_slice[bb[0]:bb[2], bb[1]:bb[3]] = current_slice[bb[0]:bb[2], bb[1]:bb[3]] | prop.filled_image
            bw[i] = current_slice

        return bw
    
    found_flag = False
    iter_count = 0
    bw0 = np.copy(bw)
    while not found_flag and iter_count < max_iter:
        label = measure.label(bw, connectivity=2)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        if len(properties) > 1 and properties[0].area/properties[1].area < max_ratio:
            found_flag = True
            bw1 = label == properties[0].label
            bw2 = label == properties[1].label
        else:
            bw = scipy.ndimage.binary_erosion(bw)
            iter_count = iter_count + 1
    
    if found_flag:
        d1 = scipy.ndimage.morphology.distance_transform_edt(bw1 == False, sampling=spacing)
        d2 = scipy.ndimage.morphology.distance_transform_edt(bw2 == False, sampling=spacing)
        bw1 = bw0 & (d1 < d2)
        bw2 = bw0 & (d1 > d2)
                
        bw1 = extract_main(bw1)
        bw2 = extract_main(bw2)
        
    else:
        bw1 = bw0
        bw2 = np.zeros(bw.shape).astype('bool')
        
    bw1 = fill_2d_hole(bw1)
    bw2 = fill_2d_hole(bw2)
    bw = bw1 | bw2

    return bw1, bw2, bw

def step1_python(case_path):
    case = load_scan(case_path)
    case_pixels, spacing = get_pixels_hu(case)
    #make image square, three options:
    #pad/trim to 512
    #pad to match longer edge
    #trim to match shorter edge
    """
    #For padding image to match longer edge
    if case_pixels.shape[1]>case_pixels.shape[2]:
        #pad shape[2] to make square
        #assume shape data is integer
        pad_slice=case_pixels.shape[1]-case_pixels.shape[2]
        half_pad1=pad_slice/2
        half_pad2=pad_slice-half_pad1
        case_pixels=np.pad(case_pixels, ((0,0),(0,0), (half_pad1,half_pad2)), 'constant', constant_values=(-1024,-1024))
    elif case_pixels.shape[2]>case_pixels.shape[1]:
        #pad shape[2] to make square
        pad_slice=case_pixels.shape[2]-case_pixels.shape[1]
        half_pad1=pad_slice/2
        half_pad2=pad_slice-half_pad1
        case_pixels=np.pad(case_pixels, ((0,0), (half_pad1,half_pad2),(0,0)), 'constant', constant_values=(-1024,-1024))
    """
    #For trimming image to match shorter edge
    shape1=case_pixels.shape[1]
    shape2=case_pixels.shape[2]
    if shape1>shape2:
        #pad shape[2] to make square
        #assume shape data is integer
        pad_slice=shape1-shape2
        half_pad1=pad_slice/2
        half_pad2=pad_slice-half_pad1
        for _ in range(half_pad2):
            case_pixels=np.delete(case_pixels, (half_pad1+shape2), axis=1)
        for _ in range(half_pad1):
            case_pixels=np.delete(case_pixels, (0), axis=1)
    elif shape2>shape1:
        #pad shape[2] to make square
        pad_slice=shape2-shape1
        half_pad1=pad_slice/2
        half_pad2=pad_slice-half_pad1
        for _ in range(half_pad2):
            case_pixels=np.delete(case_pixels, (half_pad1+shape1), axis=2)
        for _ in range(half_pad1):
            case_pixels=np.delete(case_pixels, (0), axis=2)
    bw = binarize_per_slice(case_pixels, spacing)
    #time vs efficiency vs range to try
    #for each iteration, cut slices by cut_step
    #try for 15 iterations
    cut_step=5
    iterations=15
    for i in range(iterations):
        if i==0:
            bwcp=np.copy(bw)
        elif(i*cut_step<len(bw)):
            pad=i*cut_step
            #print(pad)
            bwcp=bw[pad:]
            #print('cut',bwcp.shape)
            bwcp=np.pad(bwcp, ((pad, 0), (0,0), (0,0)), 'edge')
            #print('pad',bwcp.shape)
        else:
            break
        bwcp, flag = all_slice_analysis(bwcp, spacing, cut_num=0)
        #print(i,flag)
        if(flag!=0):
            break
    if(flag==0):
        raise Exception('Unable to segment image')
    bw = fill_hole(bwcp)
    bw1, bw2, bw = two_lung_only(bw, spacing)
    return case_pixels, bw1, bw2, spacing
    

