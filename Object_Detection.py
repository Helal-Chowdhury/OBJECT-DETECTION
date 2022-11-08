#Author: Helal Chowdhury
#Version: 1
from __future__ import print_function, division
#import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import time
import os
import itertools
import glob
import requests
import math
import shutil
from getpass import getpass
from PIL import Image, UnidentifiedImageError,ImageOps
from requests.exceptions import HTTPError
from io import BytesIO


import streamlit as st

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #0099ff;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #00ff00;
    color:#ff0000;
    }
</style>""", unsafe_allow_html=True)



#--------------------------------

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pathlib
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

import object_detection

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
st.header("Object Detection")

def load_model(model_name):

  base_url = 'http://download.tensorflow.org/models/OBJECT/'

  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"

  model = tf.saved_model.load(str(model_dir))

  return model
  
  
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'object_detection/data/mscoco_label_map.pbtxt'
#PATH_TO_LABELS = '/home/helal/Desktop/OBJECT/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)  

#PATH_TO_TEST_IMAGES_DIR = pathlib.Path('test_images')
#print("TEST IMAGE PATH",PATH_TO_TEST_IMAGES_DIR)
#TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
#print(TEST_IMAGE_PATHS)

model_name = 'ssd_mobilenet_v1_coco_2017_11_17'

with st.spinner('Wait for it...'):
    detection_model = load_model(model_name)
    #time.sleep(10)
st.success('Done!')


#detection_model.signatures['serving_default'].output_dtypes


def run_inference_for_single_image(model, image):

  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict
  

img_placeholder = st.empty() 

def show_inference(model, image_path):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = np.array(Image.open(image_path))
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  image_np=vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)
      #return image_np

  #display(Image.fromarray(image_np))
  # Display the image with the detections in the Streamlit app
  img_placeholder.image(image_np)
  
uploaded_file=st.sidebar.file_uploader("Upload image",type="jpg")
generate_pred=st.sidebar.button("Detect Objects")

if generate_pred:
	#model=get_model(model_name) 
	#image=Image.open(uploaded_file)
	image=uploaded_file
	show_inference(detection_model, image)
  
