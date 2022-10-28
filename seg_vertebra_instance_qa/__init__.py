# default

import os

if os.path.isdir(r'S:\Data\VerSe_full\nnunet\output\281'):
    dataframe_file_path = r'S:\Data\VerSe_full\nnunet\output\281'
    dataframe_file = r'seg_instance_masks_dataframe_manuel.csv'
    dataframe_file_4classes = r'df_instance_4_classes_label_manuel_partition_bbox.csv'
    dataframe_file_4classes = r'df_instance_4_classes_label_5_manuel.csv'
    data_path = r'S:\Data\VerSe_full\nnunet\output\281\segmentation_instance_masks'
    multi_gpu_flag = 0
    
elif os.path.isdir(r'/home/sukin699/Verse/Verse_full/nnunet/output/281'):
    dataframe_file_path = r'/home/sukin699/Verse/Verse_full/nnunet/output/281'
    dataframe_file = r'seg_instance_masks_dataframe_manuel.csv'
    dataframe_file_4classes = r'df_instance_4_classes_label_manuel_partition_bbox.csv'
    dataframe_file_4classes = r'df_instance_4_classes_label_5_manuel.csv'
    data_path = r'/home/sukin699/Verse/Verse_full/nnunet/output/281/segmentation_instance_masks'
    multi_gpu_flag = 1
    
else:
    dataframe_file_path = input('dataframe_file_path')
    dataframe_file = input('dataframe_filename')
    dataframe_file_4classes = input('dataframe_file_4classes')
    data_path = input('data_path')
    multi_gpu_flag = int(input('Multi GPU? 0 = No, 1 = Yes'))