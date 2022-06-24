import glob
import numpy as np
import os
import pandas as pd
import cv2
import csv


csv_file_dir = './preprocess_data/ffhq_data/ffhq_aging_labels.csv'
# img_base_dir = './preprocess_data/imdb_data/imdb_crop/'
# names = ['genders, ages, img_paths']

# file = pd.read_csv(csv_file_dir, sep =',', names=names)
# ffhq names -> image_number,age_group,age_group_confidence,gender,gender_confidence,head_pitch,head_roll,head_yaw,left_eye_occluded,right_eye_occluded,glasses

with open(csv_file_dir, 'r') as file:
    csv_file = csv.reader(file)
    
    # gender 0 -> female  / gender 1 -> male
    for line in csv_file:
        
        file_name = line[0]
        age_group = line[1]
        gender_group = line[3]
        # print(age_group)
        if age_group == '0-2':
            age = 0
        elif age_group == '3-6':
            age = 0
        elif age_group == '7-9':
            age = 0
        elif age_group == '10-14':
            age = 1
        elif age_group == '15-19':
            age = 1
        elif age_group == '20-29':
            age = 2
        elif age_group == '30-39':
            age = 3
        elif age_group == '40-49':
            age = 4
        elif age_group == '50-69':
            age = 5
        else:
            age = 6
        
        if gender_group == 'female':
            gender = 0
        else:
            gender = 1

        file_list_idx = int(file_name) // 1000
        file_list_name = str(file_list_idx).zfill(2)+ '000'
        # img_path = os.path.join(file_list_name, file_name)
        file_name = file_name.zfill(5)
        print(file_name)
        img_path = file_list_name + '/' + file_name
        img_path += '.png'
        print(img_path)
        
        # img_path, gender, age

        
        # gender, age, img_paths = line


