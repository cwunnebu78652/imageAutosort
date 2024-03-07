import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
import flyr

def compareMasks(directory, originalMaskFilename):
    try:
        # Read the images of the masks from the directory
        masks = readMaskImages(directory)

        # Read the image of the original mask
        original_mask_image = readMaskImage(originalMaskFilename)

        # Compare each mask with the original mask
        matches = [compareWithOriginal(mask_image, original_mask_image) for mask_image in masks]

        # Return a list of boolean values indicating whether each mask matches the original mask
        return matches

    except Exception as e:
        print(f"Error: {e}")
        return []


def readMaskImages(directory):
    # Read all cattle images from the specified directory using os.walk
    images = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):  # Adjust the file formats as needed
                images.append(os.path.join(root, file))
    return images


def readMaskImage(filepath):
    # Read the image of a mask file using OpenCV
    # Update the read mode based on the file format
    # For flexibility, allow parameterization based on the file format
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return image


def compareWithOriginal(mask_image, original_mask_image):
    # Use structural similarity index (SSIM) for more robust comparison
    # SSIM values range from -1 to 1, where 1 means a perfect match
    ssim_index = ssim(mask_image, original_mask_image)

    # You can adjust the threshold based on your specific requirements
    threshold = 0.95
    return ssim_index > threshold


# Example usage:
directory_path = r'C:\Users\w12j692\Desktop\MainFileCattle\all_cattle'
original_mask_folder = r'C:\Users\w12j692\Desktop\MainFileCattle\HandsortedCattle\Sorted\train\background'
output_path = r'C:\Users\w12j692\Desktop\MainFileCattle\HandsortedCattle\SortedGrayCows\train\background'

os.makedirs(output_path, exist_ok=True)
cattle_images = readMaskImages(directory_path)

files = os.listdir(original_mask_folder)

for original_mask_filename in files:

    original_filename = original_mask_filename.split('\\')[-1]
    if 'FLIR' in original_filename:
        index = original_filename.index('_')
        cattle_file = original_filename[:index] + '.jpg'
    else:
        indices = [i for i in range(len(original_filename)) if original_filename[i] == '_']
        if len(indices) < 5:
            cattle_file = original_filename[:indices[3]] + '.jpg'
        else:
            cattle_file = original_filename[:indices[4]] + '.jpg'

    if 'optical' in original_filename:
        band = 'optical'
    else:
        band = 'thermal'

    for f in cattle_images:
        if cattle_file in f:
            org_file = f
            break

    mask = cv2.imread(os.path.join(original_mask_folder, original_mask_filename), cv2.IMREAD_GRAYSCALE)
    thermogram = flyr.unpack(org_file)

    if band == 'optical':
        img = thermogram.optical
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    else:
        img = thermogram.kelvin

    msk_img = cv2.resize(mask, (img.shape[1], img.shape[0]))
    msk_img = msk_img > 0.5 * 255
    img[~msk_img] = 0
    img = (img - np.min(img[msk_img])) / (np.max(img) - np.min(img[msk_img]))
    img[~msk_img] = 0

    img = (img*255).astype('uint8')

    cv2.imwrite(os.path.join(output_path, original_filename), img)


    #result_matches = compareMasks(directory_path, original_mask_filename)

    # Print results for each mask
    #for i, match in enumerate(result_matches):
    #    print(f"Mask {i + 1} matches original mask: {match}")
