# default

import os

if os.path.isdir(r'S:\Data\VerSe_full\nnunet\output\281'):
    dataframe_file_path = r'S:\Data\VerSe_full\nnunet\output\281'
    dataframe_file = r'seg_instance_masks_dataframe_manuel.csv'
    data_path = r'S:\Data\VerSe_full\nnunet\output\281\segmentation_instance_masks'
elif os.path.isdir(r'/home/sukin699/Verse/Verse_full/nnunet/output/281'):
    dataframe_file_path = r'/home/sukin699/Verse/Verse_full/nnunet/output/281'
    dataframe_file = r'seg_instance_masks_dataframe_manuel.csv'
    data_path = r'/home/sukin699/Verse/Verse_full/nnunet/output/281/segmentation_instance_masks'