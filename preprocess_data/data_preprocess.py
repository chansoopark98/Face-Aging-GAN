import glob
import numpy as np
import os
import pandas as pd
import cv2
import csv


csv_file_dir = './preprocess_data/imdb_data/imdb.csv'
img_base_dir = './preprocess_data/imdb_data/imdb_crop/'
# names = ['genders, ages, img_paths']

# file = pd.read_csv(csv_file_dir, sep =',', names=names)

f = open(csv_file_dir, 'r')

rdr = csv.reader(f)

for line in rdr:
    print(line[2])                

f.close()

with open(csv_file_dir, 'r') as file:
    csv_file = csv.reader(file)
    
    for line in csv_file:
        gender, age, img_paths = line


