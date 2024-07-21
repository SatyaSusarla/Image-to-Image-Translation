import os
import cv2
import numpy as np
import random

def create_negative(image):
    return 255 - image

def apply_orange_mask(image, intensity=0.3):
    orange_mask = np.zeros_like(image, dtype=np.uint8)
    orange_mask[:] = (0, 165, 255)  # BGR for orange
    blended_image = cv2.addWeighted(image, 1 - intensity, orange_mask, intensity, 0)
    return blended_image

def apply_general_old_film_effect(image, orange_intensity=0.2):
    # Add subtle grain
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)  # Further reduced intensity for grain
    grainy_image = cv2.add(image, noise)

    # Add a more subtle sepia tone
    kernel = np.array([[0.393, 0.769, 0.189],
                       [0.349, 0.686, 0.168],
                       [0.272, 0.534, 0.131]])
    sepia_image = cv2.transform(grainy_image, kernel)

    # Clip values to maintain valid range
    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)

    # Apply a small orange tint
    final_image = apply_orange_mask(sepia_image, intensity=orange_intensity)  # Adjustable orange tint

    return final_image

def apply_very_old_film_effect(image):
    # Add more intense grain
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)  # Increased intensity for grain
    grainy_image = cv2.add(image, noise)

    # Add sepia tone
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(grainy_image, kernel)

    # Clip values to maintain valid range
    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)

    # Apply a small orange tint
    final_image = apply_orange_mask(sepia_image, intensity=0.1)  # Subtle orange tint

    return final_image

def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    jpg_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]
    total_files = len(jpg_files)

    num_simple_negative = int(0.15 * total_files)
    num_orange_mask = int(0.60 * total_files)
    num_general_old_film = int(0.20 * total_files)
    num_very_old_film = total_files - num_simple_negative - num_orange_mask - num_general_old_film

    # Randomly select files for each category
    simple_negative_files = random.sample(jpg_files, num_simple_negative)
    remaining_files = [f for f in jpg_files if f not in simple_negative_files]
    orange_mask_files = random.sample(remaining_files, num_orange_mask)
    remaining_files = [f for f in remaining_files if f not in orange_mask_files]
    general_old_film_files = random.sample(remaining_files, num_general_old_film)
    very_old_film_files = [f for f in remaining_files if f not in general_old_film_files]

    for file_name in jpg_files:
        img_path = os.path.join(input_dir, file_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Could not read image {file_name}")
            continue

        negative_img = create_negative(img)

        if file_name in simple_negative_files:
            final_img = negative_img
        elif file_name in orange_mask_files:
            final_img = apply_orange_mask(negative_img)
        elif file_name in general_old_film_files:
            final_img = apply_general_old_film_effect(negative_img, orange_intensity=0.2)
        elif file_name in very_old_film_files:
            final_img = apply_very_old_film_effect(negative_img)

        # Change the file extension to PNG
        base_name = os.path.splitext(file_name)[0]
        output_file_name = f"{base_name}.png"
        output_path = os.path.join(output_dir, output_file_name)

        cv2.imwrite(output_path, final_img)
        print(f"Processed and saved: {output_path}")

input_directory = '/Users/satyasusarla/code/CNN-Project/Data-Photos'  # Change to your input directory
output_directory = '/Users/satyasusarla/code/CNN-Project/Final-Photos'

process_images(input_directory, output_directory)

