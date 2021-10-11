
# Required libraries to run this script
import argparse
import tensorflow as tf
import tensorflow_hub as hub
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
from PIL import Image

# ArgParse Function

def parse_args():
    # Create argparse object
    parser = argparse.ArgumentParser(description = 
       'A  script that predicts flowers from images using pre-trained tensorflow Hub classifier model. User supplies an                                           image file and a classifier model .h5 file. By default, the program outputs top 5 predictions and flower names but user can control the latter two             optional arguments. See --help file for details')
    
    # Add Arguments
    parser.add_argument('image_path',
                         help = "image filepath",
                         type = str)
    parser.add_argument('model_path',
                         help = ".h5 model filepath",
                         type = str)
    parser.add_argument('--top_k',
                         help = "Top K predictions. By default K = 5 but user can set other integer values",
                         type =int,
                         default =5)
    parser.add_argument('--category_names',
                         help = "A json dictionary mapping flower labels to their names. A default dictionary is used but user can also supply his json file",
                         type = str,
                         default = 'label_map.json')
    return parser.parse_args()

# FUnction to process image
def process_image(image_path):
    image = tf.convert_to_tensor(image_path)         # convert image numpy object to tensorflow object
    image = tf.cast(image, tf.float32) 
    image = tf.image.resize(image, (224, 224))   # reshape image to 224 x 224
    image = image/255                            # normalize pixels (0-255) to (0-1)
    image = image.numpy()                        # convert tf object to numpy array
    return image

# Read in flowers dictionary
def label_map(category_names):
  with open(category_names, 'r') as f:
    labels = json.load(f)
    return labels
    
# Predict function
def predict(image_path, model_path, top_k):

    #Load model
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer}, compile = False)  

    # Load image using PIL Image method
    im = Image.open(image_path)
    im = np.asarray(im)  # convert image to numpy array

    # Process image using process_image function. Outputs image with shape (224, 224, 3 )
    im = process_image(im)    

    # Make prediction using trained model
    im = np.expand_dims(im, axis =0)   # convert image to (0,224,224,3)
    predictions = model.predict(im).squeeze()   # Prediction is in 1D numpy array of probabilities

    ## Retrieve Top N classes and their probabilities 
    indices = np.argsort(-predictions)[:top_k]        
    classes = indices + 1
    probs = predictions[indices]
    # names = [class_names[str(i)] for i in classes]
    return classes, probs

def main():
    args = parse_args()
    image_path = args.image_path
    model_path = args.model_path
    top_k = args.top_k
    category_names = args.category_names
    
    classes, probs = predict(image_path, model_path, top_k)
    
    flowers = [label_map(category_names)[str(i)] for i in classes]
    
    print("Your inputs:", vars(args))
    print("TOP " + str(top_k) +" PREDICTIONS")
    print("Flower Labels: ", list(classes))
    print("Probabilities: ", list(np.round(probs,3)))
    print("Flowers: ", flowers)
        
if __name__ == "__main__":
    main()