import numpy as np
from skimage.draw import polygon
import pydicom
import os
import argparse
import subprocess
import time
import nibabel as nib
import re


def readDICOMandRTSS(folder_path):   
    # 讀取DICOM以及對應RTSS檔案，RTSS檔案會在DICOM資料夾內，且只有一個

    dicom_files = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:            
            dcm_file = pydicom.dcmread(os.path.join(dirpath, filename))

            if dcm_file.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.3': # SOPClassUID for RTSS type file
                rtss = dcm_file
            else:
                dicom_files.append(dcm_file)

    dicom_images = []
    slice_position = []
    SeriesInstanceUID = dicom_files[0].SeriesInstanceUID
    for dicom_file in dicom_files:
        assert SeriesInstanceUID == dicom_file.SeriesInstanceUID # 確保append的都是同一個series
        slice_position.append(dicom_file.ImagePositionPatient[-1])
        dicom_images.append(dicom_file.pixel_array)

    

    # sort by slice position
    sort_index = np.argsort(slice_position)
    dicom_files = [dicom_files[i] for i in sort_index]
    slice_position = [slice_position[i] for i in sort_index]
    dicom_images = [dicom_images[i] for i in sort_index]
    dicom_images = np.stack(dicom_images)
    dicom_images = dicom_images.transpose(1,2,0) # HxWxD


    if 'rtss' in locals():
        return dicom_files, dicom_images, rtss
    else:
        return dicom_files, dicom_images
    

def clean_filename(filename):
    # Windows非法字符
    illegal_chars_windows = r'[<>:"/\\|?*]'
    # Unix/Linux非法字符
    illegal_chars_unix = r'[/\0]'

    # 移除非法字符
    cleaned_filename = re.sub(illegal_chars_windows, '', filename)
    cleaned_filename = re.sub(illegal_chars_unix, '', cleaned_filename)

    return cleaned_filename


def read_structure(structure):
    # 將contour資訊獨立出來

    contours = []
    for i in range(len(structure.ROIContourSequence)):
        contour = {}
        contour['color'] = structure.ROIContourSequence[i].ROIDisplayColor
        contour['number'] = structure.ROIContourSequence[i].ReferencedROINumber
        contour['name'] = structure.StructureSetROISequence[i].ROIName
        assert contour['number'] == structure.StructureSetROISequence[i].ROINumber
        contour['contours'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
        contour['ReferencedSOPInstanceUID'] = [s.ContourImageSequence[0].ReferencedSOPInstanceUID \
                                                for s in structure.ROIContourSequence[i].ContourSequence] #每組contour對應的切面
        contours.append(contour)

    return contours



def make_affine_mat(dicom_file, inv=False):
    # 將affine矩陣取出，並計算對應反矩陣
    # 相關座標換算可參考: https://www.slicer.org/wiki/Coordinate_systems

    # 提取方向余弦
    row_cosines = np.array(dicom_file.ImageOrientationPatient[0:3])
    col_cosines = np.array(dicom_file.ImageOrientationPatient[3:6])
    
    # 構建仿射矩陣
    affine_mat = np.zeros((4, 4))
    affine_mat[0:3, 0] = row_cosines * dicom_file.PixelSpacing[0]
    affine_mat[0:3, 1] = col_cosines * dicom_file.PixelSpacing[1]
    affine_mat[0:3, 2] = [0, 0, dicom_file.SliceThickness]
    affine_mat[0:3, 3] = dicom_file.ImagePositionPatient
    affine_mat[3, 3] = 1
    
    if inv:
        affine_mat_inv = np.linalg.inv(affine_mat)
        affine_dict = {'mat':affine_mat, 'inv':affine_mat_inv}
    else:
        affine_dict = {'mat':affine_mat}

    return affine_dict


def getlabelmask(dicom_files,dicom_images,contours,save_in_split=False):
    # 取得label mask

    # 建立對應切面資訊
    z = [s.ImagePositionPatient[2] for s in dicom_files]
    SOPInstanceUID = [s.SOPInstanceUID for s in dicom_files]
    affine_MAT = [make_affine_mat(info, inv=True) for info in dicom_files]

    # 建立mask
    if save_in_split:
        label_msk = np.zeros((*dicom_images.shape,len(contours)), dtype=np.uint8)
    else:
        label_msk = np.zeros_like(dicom_images, dtype=np.uint8)

    for index, con in enumerate(contours):
        num = int(con['number'])

        for i in range(len(con['contours'])):

            # 取出座標點
            c = con['contours'][i]
            nodes = np.array(c).reshape((-1, 3))
            assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0 #確保z軸座標一致
            
            # 對應z軸切面
            #z_index = z.index(nodes[0, 2])
            uid = con['ReferencedSOPInstanceUID'][i]
            z_uid_index = SOPInstanceUID.index(uid)
            #assert z_index==z_uid_index #確保以z軸座標以及UID對應方式一致

            # 根據對應切面取得Affine matrix算回IJK座標
            affine_mat_inv = affine_MAT[z_uid_index]['inv']
            ones_array = np.ones((nodes.shape[0], 1))
            nodes2 = np.concatenate((nodes, ones_array), axis=1)
            nodes_ijk = np.round(np.dot(affine_mat_inv,nodes2.transpose()).transpose())
            nodes_ijk = nodes_ijk[:,0:3]
            rr, cc = polygon(nodes_ijk[:,1], nodes_ijk[:,0])
            if save_in_split:
                label_msk[rr, cc, z_uid_index, index] = 1
            else:
                label_msk[rr, cc, z_uid_index] = num

    colors = tuple(np.array([con['color'] for con in contours]) / 255.0)

    return  label_msk, colors, affine_MAT



def adjust_orientation(img,affine_MAT,nii):
    # 調整label mask的方向與nii檔案一致

    # dicom LPS
    if affine_MAT[0]['mat'][0,0]<0:
        img = np.fliplr(img)
    if affine_MAT[0]['mat'][1,1]<0:
        img = np.flipud(img)

    # dicom LPS to RAS
    img = np.fliplr(img)
    img = np.flipud(img)
    
    # dicom RAS to nii raw
    if nii.affine[0,0]<0:
        img = np.fliplr(img)
    if nii.affine[1,1]<0:
        img = np.flipud(img)
    if nii.affine[2,2]<0:
        img = np.flip(img,axis=2)

    if len(img.shape)==3:        
        img = img.transpose((1,0,2))
    elif len(img.shape)==4:        
        img = img.transpose((1,0,2,3))

    return img


def main(**kwargs):
    # 主要執行程式
    
    #print(kwargs)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dicom_path = kwargs['dicom_path'] 
    output_path = kwargs['output_path']
    single_class = kwargs['single_class']
    save_in_split = kwargs['save_in_split']
    #print(kwargs)
    
    # run the script --------------------------   



    # read the DICOM and RTSS files
    dicom_files, dicom_images, rtss = readDICOMandRTSS(dicom_path)

    # extract the contour information
    contours = read_structure(rtss)

    # get the label mask
    label_msk, colors, affine_MAT = \
        getlabelmask(dicom_files,dicom_images,contours, save_in_split=save_in_split)
    if single_class:
        label_msk[label_msk>1] = 1

    # output the nii for the DICOM series
    exe_file = os.path.join(script_dir,'dcm2niix.exe')
    parameters = ['-b', 'n', '-z', 'n', '-f', 'img', '-o', output_path, dicom_path]
    command = [exe_file] + parameters
    subprocess.run(command, shell=True)

    # adjust for the orientation of the label mask  
    nii = nib.load(os.path.join(output_path,'img.nii'))
    label_msk = adjust_orientation(label_msk, affine_MAT, nii)

    # save the label mask to nii
    header = nii.header.copy()
    header.set_data_dtype(label_msk.dtype) 
    if save_in_split:
        for i in range(label_msk.shape[3]):
            label_mask_nii = nib.Nifti1Image(label_msk[:,:,:,i], nii.affine, header)
            nib.save(label_mask_nii, os.path.join(output_path, \
                                                'label_mask_' + clean_filename(contours[i]['name']) + '.nii.gz'))
    else:
        label_mask_nii = nib.Nifti1Image(label_msk, nii.affine, header)
        nib.save(label_mask_nii, os.path.join(output_path,'label_mask.nii.gz'))

    # save the label info
    with open(os.path.join(output_path,'label_info.txt'), 'w') as file:
        for entry in contours:
            file.write(f"Name: {entry['name']}, Number: {entry['number']}\n")
        if single_class:
            file.write("The output label mask has become single class in the nii file.")
        if save_in_split:
            file.write("The output label mask has become many nii files for each structure.")   



    # end of the run---------------------------



if __name__ == '__main__':
    
    # define the input arguments
    parser = argparse.ArgumentParser(description='Script for converting the DICOM and RTSS to the NIfTI file.')
    parser.add_argument('--dicom-path', type=str, required=True, help='Path for the dicom path (should only contain one seires with the corresponding RTSS file).')
    parser.add_argument('--output-path', type=str, help='Path for output the result NIfTI and info files.')
    parser.add_argument('--single-class', action='store_true', help='Let the output label mask become single class.')
    parser.add_argument('--save-in-split', action='store_true', help='Let the output label mask become multiple nii files for each structure.')

    # resolve the args
    args = parser.parse_args()    
    
    start_time = time.time() # start time
    # pass to the main
    main(**vars(args))
    
    end_time = time.time() # end time
    elapsed_time = end_time - start_time # elapsed time
    print("elapsed_time：{:.2f}s".format(elapsed_time))