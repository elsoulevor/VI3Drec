import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, pooling='avg')

def preproc_img(img_path, depth_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    depth_map = cv2.imread(depth_path)
    # depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
    
    img_resized = cv2.resize(img, (224, 224))
    depth_resized = cv2.resize(img, (224, 224))
    
    img_resized = img_resized / 255.0
    depth_resized = depth_resized / 255.0
    
    return img, depth_map, img_resized, depth_resized

def segment_img(model, img):
    input_tensor = np.expand_dims(img, axis=0)
    predictions = model.predict(input_tensor)
    return predictions

def focus_objects(segmentation, depth_map):
    focus_threshold = 0.5
    focused_regions = depth_map > focus_threshold
    
    contours, _ = cv2.findContours(focused_regions.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    focused_objects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        focused_objects.append((x, y, w, h))
        
    return focused_objects

def crop_objects(img, focused_objects):
    cropped_images = []
    for (x, y, w, h) in focused_objects:
        cropped_img = img[y:y+h, x:x+w]
        cropped_images.append(cropped_img)
    return cropped_images

def save_cropped_imgs(cropped_images):
    for idx, img in enumerate(cropped_images):
        cv2.imwrite(f'cropped_object_{idx}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
image_path = 'path_to_your_rgb_image.jpg'
depth_path = 'path_to_your_colormap_depth_map.png'

image, depth_map, image_resized, depth_resized = preproc_img(image_path, depth_path)
segmentation = segment_img(model, image_resized)
focused_objects = focus_objects(segmentation, depth_resized)
cropped_images = crop_objects(image, focused_objects)
save_cropped_imgs(cropped_images)