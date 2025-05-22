import os
import shutil
import random

original_data_dir = "dataset/Potato"  # existing directory
output_base_dir = "dataset"
categories = ['Early_Blight', 'Late_Blight', 'Healthy', 'Potato_Rust', 'Potato_Scab']
split_ratios = (0.7, 0.15, 0.15)  # train, val, test

def prepare_split():
    for cat in categories:
        images = os.listdir(os.path.join(original_data_dir, cat))
        random.shuffle(images)
        
        total = len(images)
        train_split = int(total * split_ratios[0])
        val_split = int(total * (split_ratios[0] + split_ratios[1]))
        
        splits = {
            'train': images[:train_split],
            'val': images[train_split:val_split],
            'test': images[val_split:]
        }

        for split_name, split_files in splits.items():
            dest_dir = os.path.join(output_base_dir, split_name, cat)
            os.makedirs(dest_dir, exist_ok=True)
            for fname in split_files:
                src = os.path.join(original_data_dir, cat, fname)
                dst = os.path.join(dest_dir, fname)
                shutil.copy2(src, dst)

if __name__ == "__main__":
    prepare_split()
    print("Dataset has been split into train, val, and test directories.")
