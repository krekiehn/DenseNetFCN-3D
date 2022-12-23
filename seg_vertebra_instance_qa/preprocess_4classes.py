import pandas as pd
import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
import random

df_gt_ct = pd.read_csv(r'S:\Data\VerSe_full\nnunet\output\281\df_instance_5.csv')
df_gt_ct = df_gt_ct.loc[df_gt_ct.gt_ct.dropna().index]

df_bbox = pd.read_csv(r'S:\Data\VerSe_full\nnunet\output\281\df_instance_bbox.csv')
df_label = pd.read_csv(r'S:\Data\VerSe_full\nnunet\output\281\df_instance_4_classes_label_manuel_partition.csv')

df = pd.read_csv(r'S:\Data\VerSe_full\nnunet\output\281\df_instance_4_classes_label_manuel_partition_bbox.csv')

def synchronize_gt_ct_bbox_label():
    bbox_list = []
    gt_ct_file_list = []
    
    for index in df_label.index:
        case_id = df_label.case_id.loc[index]
        instance_id = df_label.instance_id.loc[index]
        
        bbox = df_bbox.bbox.loc[(df_bbox.Ids == case_id) & (df_bbox.label == instance_id)].values[0]
        gt_ct_file_case_id = f"verseM_{case_id:03d}_multi_label.nii.gz"
        gt_ct_file = df_gt_ct.gt_ct.loc[(df_gt_ct.name == gt_ct_file_case_id) & (df_gt_ct.instance_index == instance_id)].values[0]
        
        bbox_list.append(bbox)
        gt_ct_file_list.append(gt_ct_file)
        print("case_id: ", case_id, "instance_id: ", instance_id, "bbox: ", bbox, "gt_ct_file: ", gt_ct_file)
    
    df_label['bbox'] = bbox_list
    df_label['gt_ct_file'] = gt_ct_file_list
    
    return df_label


def sync_peeling5_df():
    df_1 = pd.read_csv(r'S:\Data\VerSe_full\nnunet\output\281\df_instance_4_classes_label_manuel_partition_bbox_shape.csv')
    df_5 = pd.read_csv(r'S:\Data\VerSe_full\nnunet\output\281\df_instance_4_classes_label_5.csv')
    df_5_bbox = pd.read_csv(r'S:\Data\VerSe_full\nnunet\output\281\df_instance_4_classes_label_5_bbox_export.csv')
    
    df_5_new_col = ['manuel_checked', 'bbox', 'bbox_shape', 'partition', 'gt_raw_ct_file',
       'rel_path_raw_file', 'rel_path_seg_file', 'volume']
    
    df_1_use_col = ['manuel_checked', 'partition', 'label']
    df_5_bbox_use_col = ['gt_raw_ct_file', 'rel_path_raw_file', 'rel_path_seg_file', 'volume', 'bbox', 'bbox_shape']
    
    for col in df_5_new_col:
        df_5[col] = [pd.NA]*len(df_5)
    
    for index in df_1.index:
        case_id = df_1.case_id.loc[index]
        instance_id = df_1.instance_id.loc[index]
        
        index_5 = df_5[(df_5.case_id == case_id) & (df_5.instance_id == instance_id) & (df_5.peeling == 1)].index.values[0]
        
        for col in df_1_use_col:
            df_5[col].loc[index_5] = df_1[col].loc[index]

    for index in df_5_bbox.index:
        assert df_5.case_id.loc[index] == df_5_bbox.case_id.loc[index] , f"Case Ids don't match: {df_5.case_id.loc[index]}, {df_5_bbox.case_id.loc[index]}"
        assert df_5.instance_id.loc[index] == df_5_bbox.instance_id.loc[index] , f"instance_id don't match: {df_5.instance_id.loc[index]}, {df_5_bbox.instance_id.loc[index]}"
        assert df_5.peeling.loc[index] == df_5_bbox.peeling.loc[index] , f"peeling don't match: {df_5.peeling.loc[index]}, {df_5_bbox.peeling.loc[index]}"

        for col in df_5_bbox_use_col:
            df_5[col].loc[index] = df_5_bbox[col].loc[index]

    return df_1, df_5, df_5_bbox


def classes_4_peeling_5_partition():
    df_5 = pd.read_csv(r'S:\Data\VerSe_full\nnunet\output\281\df_instance_4_classes_label_5_manuel.csv')
    split = [0.8, 0.1, 0.1]
    df = df_5[df_5.partition.isna()]
    classes = [0,1,2,3]
    
    train = []
    vali = []
    test = []
    
    for cla in classes:
        indices = list(df[df.label == cla].index)
        random.shuffle(indices)
        
        train += indices[ : int(np.ceil(len(indices)*split[0]))]
        vali += indices[int(np.ceil(len(indices)*split[0])) : int(np.ceil(len(indices)*(split[1]+split[0])))]
        test += indices[int(np.ceil(len(indices)*(split[1]+split[0]))) :]
    
    #d = {train: 'train', vali: 'vali', test: 'test'}
    for pati, val in zip([train, vali, test], ['train', 'vali', 'test']):
        #val = d[pati]
        for i in pati:
            df_5.partition.loc[i] = val
    return df_5

def build_verse_snipet_bbox(verse_data_path = r'S:\Data\VerSe_full', output_data_path = r'S:\Data\VerSe_full\nnunet\output\281\segmentation_instance_masks\raw'):
    # ToDo:
    # iterate over df index
    # For each index / instance build a snipet from raw verse CT
    # find bbox, raw_ct_file
    # load raw_ct with sitk, convert it to numpy array (bbox coords belonging to that coord system)
    # extract snipet array from raw_ct_arr
    # find instance file_name and build raw_ct_snipet file_name
    # save snipet_arr as numpy array and nifti with the name format of instance file
    case_id = 0
    
    for index in tqdm(df.index):
        instance_id = df.instance_id.loc[index]
        # extract infos from df and build output file names
        bbox = eval(df.bbox.loc[index])
        # convert bbox coords tupel to a tupel of slice() operators
        bbox_slice = (slice(bbox[0], bbox[1]+1), slice(bbox[2], bbox[3]+1), slice(bbox[4], bbox[5]+1))
        # for shape shake of the raw ct bbox
        bbox_shape = (bbox[1]-bbox[0]+1, bbox[3]-bbox[2]+1, bbox[5]-bbox[4]+1)

        # Problem: file in df is seg_mask. we need the raw ct. can we convert it?
        # source: r'01_training\\derivatives\\sub-gl003\\sub-gl003_dir-ax_seg-vert_msk.nii.gz'
        # target: r'01_training\\rawdata\\sub-gl003\\sub-gl003_dir-ax_ct.nii.gz'
        
        raw_ct_file = df.gt_ct_file.loc[index]
        # does it work for all cases in verse?
        raw_ct_file = raw_ct_file.replace('derivatives', 'rawdata')
        raw_ct_file = raw_ct_file.replace('seg-vert_msk', 'ct')
        raw_ct_file_path = os.path.join(verse_data_path, raw_ct_file)

        instance_file_name = df.instance_file.loc[index]

        raw_ct_bbox_numpy_file_name = instance_file_name.split('.')[0] + '_raw.npy'
        raw_ct_bbox_numpy_file_path = os.path.join(output_data_path, raw_ct_bbox_numpy_file_name)

        raw_ct_bbox_nifti_file_name = instance_file_name.split('.')[0] + '_raw.nii'
        raw_ct_bbox_nifti_file_path = os.path.join(output_data_path, raw_ct_bbox_nifti_file_name)
        
        # load raw CT
        # for caching ct log case_id:
        if df.case_id.loc[index] != case_id:
            case_id = df.case_id.loc[index]
            ct = sitk.ReadImage(raw_ct_file_path)
            ct_arr = sitk.GetArrayFromImage(ct)
        # extract bbox from ct_arr
        ct_bbox_arr = ct_arr[bbox_slice]
        assert ct_bbox_arr.shape == bbox_shape , f"CT bbox have shape {ct_bbox_arr.shape}, but bbox shape hast to be " \
                                                 f"{bbox_shape} from bbox {bbox}. case id: {case_id}, instance id: {instance_id}, " \
                                                 f"df index: {index}"

        # save raw ct bbox as npy and nii
        np.save(raw_ct_bbox_numpy_file_path, ct_bbox_arr)

        ct_arr_bbox_nii = sitk.GetImageFromArray(ct_bbox_arr)
        ## ct_arr_bbox_nii.CopyInformation(ct)
        ct_arr_bbox_nii.SetOrigin(ct.GetOrigin())
        ct_arr_bbox_nii.SetSpacing(ct.GetSpacing())
        ct_arr_bbox_nii.SetDirection(ct.GetDirection())
        
        # source: https://discourse.itk.org/t/how-to-crop-a-3d-image-with-a-specified-size/715/5
        # cropper = sitk.ExtractImageFilter(Input=ct)
        # cropper.SetDirectionCollapseToIdentity()
        # extraction_region = cropper.GetExtractionRegion()
        # size = extraction_region.GetSize()
        # size[0] = int(bbox_shape[-1])
        # size[1] = int(bbox_shape[-2])
        # size[2] = int(bbox_shape[-3])
        # index = extraction_region.GetIndex()
        # index[0] = int(bbox[4])
        # index[1] = int(bbox[2])
        # index[2] = int(bbox[0])
        # extraction_region.SetSize(size)
        # extraction_region.SetIndex(index)
        # cropper.SetExtractionRegion(extraction_region)
        
        sitk.WriteImage(ct_arr_bbox_nii, raw_ct_bbox_nifti_file_path)
        # sitk.WriteImage(cropper, raw_ct_bbox_nifti_file_path)
        
    #return ct, ct_arr_bbox_nii
    
def shape_size_from_bbox():
    shapes = []
    for index in df.index:
        bbox = eval(df.bbox.loc[index])
        shape = (bbox[1]-bbox[0],bbox[3]-bbox[2],bbox[5]-bbox[4])
        shapes.append(shape)
    return shapes


def spacing_from_nii():
    '''Load spacing value from nifti and save it to dataframe'''
    df = pd.read_csv(r'S:\Data\VerSe_full\nnunet\output\281\df_instance_4_classes_label_5_manuel_unix_syn_all.csv')
    spacings = []
    for i in tqdm(df.index.values):
        ff = df.rel_path_seg_file.loc[i]
        ff = ff.split('.npy')[0] + '.nii'
        img_path = os.path.join(r'S:\Data\VerSe_full\nnunet\output\281', ff)
        r = sitk.ImageFileReader()
        r.SetFileName(str(img_path))
        r.LoadPrivateTagsOn()
        r.ReadImageInformation()
        spacing = r.GetSpacing()
        spacings.append(spacing)
    df['spacings'] = spacings
    return df


def uniform_spacing(desired_spacing=(1,1,1)):
    import scipy.ndimage
    
    df = pd.read_csv(r'S:\Data\VerSe_full\nnunet\output\281\df_instance_4_classes_label_5_manuel_unix_syn_all.csv')

    ff = df.rel_path_seg_file.loc[0]
    volume = np.load(os.path.join(r'S:\Data\VerSe_full\nnunet\output\281', ff))
    spacing = eval(df.spacings.loc[0])[::-1]
    print(spacing)
    
    if np.not_equal(tuple(spacing), desired_spacing).any():
        scipy_dtype = 'int16'
        # if dtype == 'float16' else dtype  # sadly scipy.ndimage.zoom does not work with float16
        volume = volume.astype(scipy_dtype, copy=False)
        volume = scipy.ndimage.zoom(volume, order=1, zoom=np.divide(spacing, desired_spacing), prefilter=False,
                                    output=scipy_dtype)
    return volume


def mosemed_check(folder_in=r'D:\Data\CT\Mosmed\mosmed', folder_out=r'D:\Data\CT\Mosmed\mosmed\mosmed'):
    files_in = os.listdir(folder_in)
    files_out = os.listdir(folder_out)

    files_todo = []

    for f in files_in:
        if not '.nii.gz' in f:
            continue
        f_tmp = f.split('_0000')[0] + f.split('_0000')[1]
        if f_tmp not in files_out:
            files_todo.append(os.path.join(files_in, f))
