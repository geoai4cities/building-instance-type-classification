# -*- coding: utf-8 -*-
# Test Predictions.ipynb
# This notebook is for scene analysis and arranging test predictions"

#@title
# for basic file navigation and driving mount
import os
from google.colab import drive

# some basic libraries
import xgboost
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# importing for basic image operations
import cv2 as cv
from PIL import Image
from PIL import ImageOps
import seaborn as sns
import pandas as pd

# data structures
from collections import OrderedDict

# importing pretrained models
from tensorflow.keras.applications import efficientnet

# for splitting data
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# prerequisites for training models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# for selecting better metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# initialize TPU
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()

drive.mount('/content/gdrive')

# variables to change

TRAIN_IMG_DIR_PATH = "/content/gdrive/Othercomputers/My Laptop/Dataset Images/instances/train/"
TEST_IMG_DIR_PATH = "/content/gdrive/Othercomputers/My Laptop/Dataset Images/instances/test/"
TRAIN_SC_DIR_PATH = "/content/gdrive/Othercomputers/My Laptop/Dataset Images/scenes/train_renamed/all_resized"
TEST_SC_DIR_PATH = "/content/gdrive/Othercomputers/My Laptop/Dataset Images/scenes/test_renamed/all_resized"
model_path = "/content/gdrive/MyDrive"
classes = ["commercial", "residential", "industrial", "others"]

INPUT_IMG_WIDTH = 250
INPUT_IMG_HEIGHT = 350
VERTICAL = True

#@title
with strategy.scope():
  new_eff_model = tf.keras.models.load_model(model_path + '/' + 'model_weights.h5')

#@title
target_class = "residential"
residential_prediction_mapping = OrderedDict()

with strategy.scope():
 
  for file_name in os.listdir(TEST_IMG_DIR_PATH + target_class):

    img = Image.open(TEST_IMG_DIR_PATH + target_class + '/'+ file_name)
    building_image = np.array(ImageOps.fit(img,(250, 350), Image.ANTIALIAS))
    building_image = np.expand_dims(building_image, axis = 0)
    residential_prediction_mapping[target_class + '/'+ file_name] = new_eff_model.predict(building_image)

target_class = "commercial"
commercial_prediction_mapping = OrderedDict()
with strategy.scope():
  
  for file_name in os.listdir(TEST_IMG_DIR_PATH + target_class):
    img = Image.open(TEST_IMG_DIR_PATH + target_class + '/'+ file_name)
    building_image = np.array(ImageOps.fit(img,(250, 350), Image.ANTIALIAS))
    building_image = np.expand_dims(building_image, axis = 0)
    commercial_prediction_mapping[target_class + '/'+ file_name] = new_eff_model.predict(building_image)

target_class = "industrial"
industrial_prediction_mapping = OrderedDict()
with strategy.scope():
  
  for file_name in os.listdir(TEST_IMG_DIR_PATH + target_class):
    img = Image.open(TEST_IMG_DIR_PATH + target_class + '/'+ file_name)
    building_image = np.array(ImageOps.fit(img,(250, 350), Image.ANTIALIAS))
    building_image = np.expand_dims(building_image, axis = 0)
    industrial_prediction_mapping[target_class + '/'+ file_name] = new_eff_model.predict(building_image)

target_class = "others"
others_prediction_mapping = OrderedDict()
with strategy.scope():
  
  for file_name in os.listdir(TEST_IMG_DIR_PATH + target_class):
  
    img = Image.open(TEST_IMG_DIR_PATH + target_class + '/'+ file_name)
    building_image = np.array(ImageOps.fit(img,(250, 350), Image.ANTIALIAS))
    building_image = np.expand_dims(building_image, axis = 0)
    others_prediction_mapping[file_name] = new_eff_model.predict(building_image)

#@title
others_predictions = OrderedDict()

for file_name in others_prediction_mapping.keys():
  # others_predictions[file_name] = np.argmax(others_prediction_mapping[file_name], axis = 1)[0]
  others_predictions[file_name] = others_prediction_mapping[file_name]

others_scene_predictions = OrderedDict()
for file_name in others_predictions.keys():
  scene_file_name = '_'.join(os.path.splitext(file_name)[0].split('_')[-3:]) + os.path.splitext(file_name)[1]
  if scene_file_name not in others_scene_predictions.keys():
    others_scene_predictions[scene_file_name] = [others_predictions[file_name]]
  else:
    others_scene_predictions[scene_file_name].append(others_predictions[file_name])

residential_predictions = OrderedDict()
for file_name in residential_prediction_mapping.keys():
  # residential_predictions[file_name] = np.argmax(residential_prediction_mapping[file_name], axis = 1)[0]
  
  residential_predictions[file_name] = residential_prediction_mapping[file_name]

residential_scene_predictions = OrderedDict()
for file_name in residential_predictions.keys():
  scene_file_name = '_'.join(os.path.splitext(file_name)[0].split('_')[-3:]) + os.path.splitext(file_name)[1]
  if scene_file_name not in residential_scene_predictions.keys():
    residential_scene_predictions[scene_file_name] = [residential_predictions[file_name]]
  else:
    residential_scene_predictions[scene_file_name].append(residential_predictions[file_name])

commercial_predictions = OrderedDict()
for file_name in commercial_prediction_mapping.keys():
  # commercial_predictions[file_name] = np.argmax(commercial_prediction_mapping[file_name], axis = 1)[0]

  commercial_predictions[file_name] = commercial_prediction_mapping[file_name]
commercial_scene_predictions = OrderedDict()
for file_name in commercial_predictions.keys():
  scene_file_name = '_'.join(os.path.splitext(file_name)[0].split('_')[-3:]) + os.path.splitext(file_name)[1]
  if scene_file_name not in commercial_scene_predictions.keys():
    commercial_scene_predictions[scene_file_name] = [commercial_predictions[file_name]]
  else:
    commercial_scene_predictions[scene_file_name].append(commercial_predictions[file_name])

industrial_predictions = OrderedDict()
for file_name in industrial_prediction_mapping.keys():
  # industrial_predictions[file_name] = np.argmax(industrial_prediction_mapping[file_name], axis = 1)[0]

  industrial_predictions[file_name] = industrial_prediction_mapping[file_name]
industrial_scene_predictions = OrderedDict()
for file_name in industrial_predictions.keys():
  scene_file_name = '_'.join(os.path.splitext(file_name)[0].split('_')[-3:]) + os.path.splitext(file_name)[1]
  if scene_file_name not in industrial_scene_predictions.keys():
    industrial_scene_predictions[scene_file_name] = [industrial_predictions[file_name]]
  else:
    industrial_scene_predictions[scene_file_name].append(industrial_predictions[file_name])