# -*- coding: utf-8 -*-
# Inference.ipynb
# This is for inference

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
TRAIN_SC_DIR_PATH = "/content/gdrive/Othercomputers/My Laptop/Dataset Images/scenes/train_renamed/all"
TEST_SC_DIR_PATH = "/content/gdrive/Othercomputers/My Laptop/Dataset Images/scenes/test_renamed/all"
model_path = "/content/gdrive/MyDrive/projects/Building Classification Part 2"
classes = ["commercial", "residential", "industrial", "others"]
INPUT_IMG_WIDTH = 250
INPUT_IMG_HEIGHT = 350
VERTICAL = True

with strategy.scope():
  new_eff_model = tf.keras.models.load_model(model_path + '/' + 'model_weights.h5')

Image.open(TRAIN_IMG_DIR_PATH + 'residential' + '/'+ "2_d_r_137.png")

"""### Get prediction mappings"""

target_class = "residential"
residential_prediction_mapping = OrderedDict()
with strategy.scope():
  
  for file_name in os.listdir(TRAIN_IMG_DIR_PATH + target_class):
  
    img = Image.open(TRAIN_IMG_DIR_PATH + target_class + '/'+ file_name)
    building_image = np.array(ImageOps.fit(img,(250, 350), Image.ANTIALIAS))
    building_image = np.expand_dims(building_image, axis = 0)
    residential_prediction_mapping[target_class + '/'+ file_name] = new_eff_model.predict(building_image)

target_class = "commercial"
commercial_prediction_mapping = OrderedDict()
with strategy.scope():
  
  for file_name in os.listdir(TRAIN_IMG_DIR_PATH + target_class):
    
    img = Image.open(TRAIN_IMG_DIR_PATH + target_class + '/'+ file_name)
    building_image = np.array(ImageOps.fit(img,(250, 350), Image.ANTIALIAS))
    building_image = np.expand_dims(building_image, axis = 0)
    commercial_prediction_mapping[target_class + '/'+ file_name] = new_eff_model.predict(building_image)

target_class = "industrial"
industrial_prediction_mapping = OrderedDict()
with strategy.scope():
  
  for file_name in os.listdir(TRAIN_IMG_DIR_PATH + target_class):
    
    img = Image.open(TRAIN_IMG_DIR_PATH + target_class + '/'+ file_name)
    building_image = np.array(ImageOps.fit(img,(250, 350), Image.ANTIALIAS))
    building_image = np.expand_dims(building_image, axis = 0)
    industrial_prediction_mapping[target_class + '/'+ file_name] = new_eff_model.predict(building_image)

target_class = "others"
others_prediction_mapping = OrderedDict()
with strategy.scope():
  
  for file_name in os.listdir(TRAIN_IMG_DIR_PATH + target_class):
  
    img = Image.open(TRAIN_IMG_DIR_PATH + target_class + '/'+ file_name)
    building_image = np.array(ImageOps.fit(img,(250, 350), Image.ANTIALIAS))
    building_image = np.expand_dims(building_image, axis = 0)
    others_prediction_mapping[file_name] = new_eff_model.predict(building_image)

others_prediction_mapping

"""### Prediction through Neighborhood Data"""

df = pd.read_csv("/content/gdrive/MyDrive/frequency_df.csv").rename(columns = {"Unnamed: 0" : "filename"})
df.head(5)



# remove all detected buildings/part of buildings
df.drop(["building", "house", "skyscraper", "tower"], axis = 1, inplace = True)
df.drop(["window"], axis = 1, inplace = True)

# let us check if there is any pattern that can segregate the classes
df["filename"] = df["filename"].apply(lambda x : "Delete" if "commercial_test" in x else x)
df = df.drop(df[df["filename"] == "Delete"].index, axis = 0)

# split the folders (important)

# df["target"] = df["filename"].apply(lambda x: x[0])
# # df["target"] = df["filename"].apply(lambda x: x.split('/')[0])
# # df.drop("filename", axis = 1, inplace = True)

# convert target to index
# scene_classes = ["commercial", "residential", "industrial", "others", "mixed"]
# scene_building_map = {'r':1, 'c':0, 'i':2, 'o':3}
# df["target"] = df["target"].apply(lambda x: scene_building_map[x])

df.filename.filter(lambda x: "b_m" in x)

df.filename = df.filename.apply(lambda x: x if "b_m" not in x else )

# training and evaluation, keeping filename, just for convinience
train, test = train_test_split(df, test_size=0.2)

data = train.drop(['target', 'filename'], axis = 1)
label = train.target
dtrain = xgboost.DMatrix(data, label=label)

data = test.drop(['target', 'filename'], axis = 1)
label = test.target
dtest = xgboost.DMatrix(data, label=label)

param = {'max_depth': 6, 'eta': 0.01, 'objective': 'multi:softprob'}
param['num_class'] = 5
param['eval_metric'] = ['mlogloss']
evallist = [(dtest, 'eval'), (dtrain, 'train')]

num_round = 450
bst = xgboost.train(param, dtrain, num_round, evallist)

bst.save_model('0001.model')

df.shape

data = df.drop(['filename', 'target'], axis = 1)
dtest = xgboost.DMatrix(data)
ypred = bst.predict(dtest)

plt.figure(figsize = (6, 8))
sns.histplot(np.argmax(ypred, axis = 1), discrete = classes)
plt.show()

from sklearn.metrics import accuracy_score
accuracy_score(df.target.values, np.argmax(ypred, axis = 1))
# 64% accuracy not bad at all!!!

ypred.shape

test_commercial_scene_predictions = commercial_scene_predictions.copy()

df['s_filename'] = df['filename'].apply(lambda x: x.split('/')[1])

for scene_image_file_name in test_commercial_scene_predictions.keys():
  print(scene_image_file_name)
  print(ypred[df['s_filename']==scene_image_file_name], test_commercial_scene_predictions[scene_image_file_name])

  test_commercial_scene_predictions[scene_image_file_name] += ypred[df['s_filename']==scene_image_file_name]





"""### Get predictions from mappings"""

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

others_scene_predictions

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

residential_scene_predictions

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

commercial_scene_predictions

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

industrial_scene_predictions





"""# Mixing"""

list_of_predictions = [commercial_scene_predictions, residential_scene_predictions, industrial_scene_predictions, others_scene_predictions]

commercial_scene_predictions

# get all the accuracy_metrics
# what is the scene label??

predictions_without_neighborhood_data = []
true_labels = []
for scene_index, scene_label in enumerate(classes):
  for image_name in list_of_predictions[scene_index].keys():
    true_labels.append(os.path.splitext(image_name)[0].split('_')[1])
    # true_labels.append(scene_index)
    predictions_without_neighborhood_data.append(list_of_predictions[scene_index][image_name])

predictions_without_neighborhood_data
label_predictions_without_neighborhood_data = []
for scene_building_predictions in predictions_without_neighborhood_data:
  if np.unique(scene_building_predictions).shape[0] == 1:
    label_predictions_without_neighborhood_data.append(scene_building_predictions[0])
    # label_predictions_without_neighborhood_data.append(np.argmax(np.bincount(scene_building_predictions)))
  else:
    label_predictions_without_neighborhood_data.append(4)
    
    # label_predictions_without_neighborhood_data.append(np.argmax(np.bincount(scene_building_predictions)))

for index, i in enumerate(true_labels):
  if i == 'm':
    true_labels[index] = 4
  elif i == 'c':
    true_labels[index] = 0
  elif i == 'r':
    true_labels[index] = 1
  elif i == 'i':
    true_labels[index] = 2
  else:
    true_labels[index] = 3

from sklearn.metrics import confusion_matrix
# generate plots
plt.figure(figsize = (12, 10))
sns.heatmap(confusion_matrix(true_labels, label_predictions_without_neighborhood_data), annot = True, cmap = "mako_r", fmt='g', xticklabels=["commercial", "residential", "industrial", "others", "mixed"], yticklabels=["commercial", "residential", "industrial", "others", "mixed"])
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.title("Confusion Matrix for Scene Classification Model Evaluation with Training Data (without considering neighborhood)")
plt.show()



confusion_matrix(true_labels, label_predictions_without_neighborhood_data)

print("Train Classification Report:\n\n")
print(classification_report(true_labels, label_predictions_without_neighborhood_data))
print("\n\n","-"*50)

"""#OTher stuff"""

xgboost.plot_importance(bst, max_num_features = 20, height = 0.3)

gbdt_predictions = np.argmax(ypred, axis = 1)

df.target

commercial_prediction_mapping

# from sklearn.manifold import TSNE
# import seaborn as sns

# tsne = TSNE(n_iter = 300000, perplexity = 50)
# result = tsne.fit_transform(df.drop("target", axis = 1))
# plt.figure(figsize = (15, 10))
# sns.scatterplot(x = result.T[0], y = result.T[1], hue = df["target"].apply(lambda x: scene_classes[x]))
# plt.show()