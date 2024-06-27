import numpy as np
from numpy.linalg import norm
import pickle
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D

import cv2
import pytesseract
from PIL import Image

def model_picker():
    name="resnet"
    if (name == 'vgg16'):
        model = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(224, 224, 3),
                      pooling='max')
    elif (name == 'vgg19'):
        model = VGG19(weights='imagenet',
                      include_top=False,
                      input_shape=(224, 224, 3),
                      pooling='max')
    elif (name == 'mobilenet'):
        model = MobileNet(weights='imagenet',
                          include_top=False,
                          input_shape=(224, 224, 3),
                          pooling='max',
                          depth_multiplier=1,
                          alpha=1)
    elif (name == 'inception'):
        model = InceptionV3(weights='imagenet',
                            include_top=False,
                            input_shape=(224, 224, 3),
                            pooling='max')
    elif (name == 'resnet'):
        model = ResNet50(weights='imagenet',
                         include_top=False,
                         input_shape=(224, 224, 3),
                        pooling='max')
    elif (name == 'xception'):
        model = Xception(weights='imagenet',
                         include_top=False,
                         input_shape=(224, 224, 3),
                         pooling='max')
    else:
        print("Specified model not available")
    return model



def extract_features(img_path, model):
    input_shape = (224, 224, 3)
    img = image.load_img(img_path,
                         target_size=(input_shape[0], input_shape[1]))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(flattened_features)
    return normalized_features

def extract_text(img_path):
    #We then read the image with text
    images=cv2.imread(img_path)
    
    #convert to grayscale image
    gray=cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    
    #checking whether thresh or blur
    #if args["pre_processor"]=="thresh":
    cv2.threshold(gray, 0,255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)[1]
    #if args["pre_processor"]=="blur":
    #    cv2.medianBlur(gray, 3)
        
    #memory usage with image i.e. adding image to memory
    filename = "{}.jpg".format(os.getpid())
    cv2.imwrite(filename, gray)
    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    return text

extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG','.ppm']

def get_file_list(root_dir):
    file_list = []
    for root, directories, filenames in os.walk(root_dir):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            if any(ext in filename for ext in extensions):
                if os.path.exists(filepath):
                  file_list.append(filepath)
                else:
                  print(filepath)
            elif "pdf" in filename:
                if not os.path.exists(f"./index/{filename}"):
                    os.mkdir(f"./index/{filename}")
                print(f"pdfimages {filepath} ./index/{filename}/")
                os.system(f"pdfimages {filepath} ./index/{filename}/")
    for root, directories, filenames in os.walk("./index"):
        for filename in filenames:
            if any(ext in filename for ext in extensions):
                filepath = os.path.join(root, filename)
                if os.path.exists(filepath):
                  file_list.append(filepath)
                else:
                  print(filepath)
    return file_list

if __name__ == "__main__":
    model = model_picker()
    root_dir = './images/'
    filenames = sorted(get_file_list(root_dir))
    print(len(filenames))

    standard_feature_list = []
    text_list=[]
    for i in range(len(filenames)):
        standard_feature_list.append(extract_features(filenames[i], model))
        text=extract_text(filenames[i])
        print(text)
        text_list.append(text)
        print(f"{i}/{len(filenames)}")

    print("Num images   = ", len(standard_feature_list))

    pickle.dump(filenames, open('./index/filenames.pickle', 'wb'))
    pickle.dump(text_list, open('./index/text.pickle', 'wb'))
    pickle.dump(
        standard_feature_list,
        open('./index/features'+ '.pickle', 'wb'))
    

    