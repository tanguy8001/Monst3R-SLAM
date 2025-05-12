import os
import argparse
from PIL import Image
from tqdm import tqdm

def convert_png_to_jpg(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    png_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.png')])

    for index, filename in enumerate(tqdm(png_files, desc="Converting")):
        if index > 100:
            break
        input_path = os.path.join(input_dir, filename)
        output_filename = f"{index}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        try:
            with Image.open(input_path) as img:
                rgb_img = img.convert('RGB')
                rgb_img.save(output_path, 'JPEG')
        except Exception as e:
            print(f"Failed to convert {filename}: {e}")

if __name__ == "__main__":
    input_dir = '/work/courses/3dv/24/MASt3R-SLAM/datasets/tum2/rgbd_dataset_freiburg3_walking_xyz/rgb'
    output_dir = '/work/courses/3dv/24/MASt3R-SLAM/logs/sam2_test_frames'
    convert_png_to_jpg(input_dir, output_dir)
