import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import requests
from tqdm import tqdm
import zipfile

# Define a simple U-Net model
def unet_model(input_size=(256, 256, 6)): # 6 because the depth map is RGB
    inputs = tf.keras.Input(input_size)
    
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    
    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    
    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = models.Model(inputs=[inputs], outputs=[outputs])
    
    return model

# Load and preprocess data
def load_data(rgb_dir, depth_dir, mask_dir, img_size=(256, 256)):
    rgb_images = []
    depth_maps = []
    masks = []
    
    for img_name in os.listdir(rgb_dir):
        rgb_img = cv2.imread(os.path.join(rgb_dir, img_name))
        depth_img = cv2.imread(os.path.join(depth_dir, img_name), cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.imread(os.path.join(mask_dir, img_name), cv2.IMREAD_GRAYSCALE)
        
        rgb_img = cv2.resize(rgb_img, img_size)
        depth_img = cv2.resize(depth_img, img_size)
        mask_img = cv2.resize(mask_img, img_size)
        
        rgb_images.append(rgb_img)
        depth_maps.append(depth_img)
        masks.append(mask_img)
    
    rgb_images = np.array(rgb_images)
    depth_maps = np.expand_dims(np.array(depth_maps), axis=-1)
    masks = np.expand_dims(np.array(masks), axis=-1)
    
    # Normalize data
    rgb_images = rgb_images / 255.0
    depth_maps = depth_maps / 255.0
    masks = masks / 255.0
    
    # Concatenate RGB and depth maps
    inputs = np.concatenate((rgb_images, depth_maps), axis=-1)
    
    return inputs, masks

# # Paths to your datasets
# rgb_dir = '/home/vinhsp/3drecpipeline/focus_segmentation/kitti_depth_completion/raw_data'
# depth_dir = 'path_to_depth_images'
# mask_dir = 'path_to_segmentation_masks'

# # Load data
# inputs, masks = load_data(rgb_dir, depth_dir, mask_dir)

# # Split data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(inputs, masks, test_size=0.2, random_state=42)

# # Create and compile the model
# model = unet_model(input_size=(256, 256, 4))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# # Evaluate the model
# loss, acc = model.evaluate(X_test, y_test)
# print(f"Test accuracy: {acc * 100:.2f}%")

# # Make predictions
# predictions = model.predict(X_test)

#################### DOWNLOADING DATASET ############################

def download_file(url, dest):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kilobyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(dest, 'wb') as file:
        for data in response.iter_content(block_size):
            t.update(len(data))
            file.write(data)
    t.close()
    
    # Verify file size
    if total_size != 0 and os.path.getsize(dest) != total_size:
        raise Exception("Downloaded file size does not match expected size.")

def extract_zip(file_path, dest_dir):
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
    except zipfile.BadZipFile:
        print(f"Error: {file_path} is not a valid zip file.")
        os.remove(file_path)
        raise


def DL_script():
    # URLs for KITTI Depth Completion dataset
    depth_data_url = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip'
    rgb_data_url = 'http://www.cvlibs.net/download.php?file=data_rgb.zip'

    # Directories
    data_dir = 'kitti_depth_completion'
    os.makedirs(data_dir, exist_ok=True)

    # Download depth data
    depth_zip_path = os.path.join(data_dir, 'data_depth_annotated.zip')
    try:
        download_file(depth_data_url, depth_zip_path)
        extract_zip(depth_zip_path, data_dir)
    except Exception as e:
        print(f"Failed to download or extract depth data: {e}")

    # Download RGB data
    rgb_zip_path = os.path.join(data_dir, 'data_rgb.zip')
    try:
        download_file(rgb_data_url, rgb_zip_path)
        extract_zip(rgb_zip_path, data_dir)
    except Exception as e:
        print(f"Failed to download or extract RGB data: {e}")

    print("Download and extraction complete.")

def download_and_extract(url, dest_path):
    local_filename = os.path.join(dest_path, url.split("/")[-1])
    # Download the file from `url` and save it locally under `local_filename`
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    # Extract the zip file
    with zipfile.ZipFile(local_filename, 'r') as zip_ref:
        zip_ref.extractall(dest_path)
    # Remove the zip file
    os.remove(local_filename)

def DL_script2():
    
    # Define the paths
    rgb_url = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip'
    mask_url = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_semantics.zip'
    data_dir = './kitti_data'

    # Create the directory if it does not exist
    os.makedirs(data_dir, exist_ok=True)

    # Download and extract the RGB images and masks
    download_and_extract(rgb_url, data_dir)
    #download_and_extract(mask_url, data_dir)

########## MAIN ###########

if __name__ == "__main__":
    DL_script2()