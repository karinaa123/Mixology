import shutil
from pathlib import Path
import os
import sys

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable


class DatasetMerger:
    def __init__(self, original_dir, synthetic_dir):
        self.original_dir = Path(original_dir)
        self.synthetic_dir = Path(synthetic_dir)

        # Use your specific folder names
        self.subsets = ['train', 'valid', 'test']
        self.supported_ext = ['.jpg', '.jpeg', '.png', '.bmp']

    def merge(self):
        print(f"Merging '{self.original_dir.name}' INTO '{self.synthetic_dir.name}'...")

        total_copied = 0

        for subset in self.subsets:
            # Source paths (Originals)
            src_img_dir = self.original_dir / subset / 'images'
            src_lbl_dir = self.original_dir / subset / 'labels'

            # Destination paths (Synthetic folder)
            dst_img_dir = self.synthetic_dir / subset / 'images'
            dst_lbl_dir = self.synthetic_dir / subset / 'labels'

            # Check if source exists
            if not src_img_dir.exists():
                print(f"Skipping {subset} (not found in source)")
                continue

            # Get list of images
            images = []
            for ext in self.supported_ext:
                images.extend(list(src_img_dir.glob(f'*{ext}')))

            print(f"\nProcessing {subset}: Merging {len(images)} original images...")

            for img_path in tqdm(images):
                # 1. Define New Name
                # Example: "image_01.jpg" -> "white_bg_image_01.jpg"
                new_filename = f"white_bg_{img_path.name}"

                # 2. Copy Image
                shutil.copy2(img_path, dst_img_dir / new_filename)

                # 3. Handle Matching Label
                label_name = img_path.stem + '.txt'
                src_label = src_lbl_dir / label_name

                if src_label.exists():
                    new_label_name = f"white_bg_{label_name}"
                    shutil.copy2(src_label, dst_lbl_dir / new_label_name)

                total_copied += 1

        print("\n" + "=" * 30)
        print("  MERGE COMPLETE")
        print("=" * 30)
        print(f"  Files copied: {total_copied}")
        print(f"  Your '{self.synthetic_dir.name}' folder now contains BOTH datasets.")
        print("=" * 30)


if __name__ == "__main__":
    # --- CONFIGURATION ---

    # 1. The folder with the White Backgrounds (Original)
    ORIGINAL_PATH = '/home/kalgaonp/anaconda3/envs/kdThesisEnv/mixology/Mixology/bottles-1'

    # 2. The folder with the Synthetic Backgrounds (Destination)
    SYNTHETIC_PATH = '/home/kalgaonp/anaconda3/envs/kdThesisEnv/mixology/Mixology/bottles_synthetic'

    merger = DatasetMerger(ORIGINAL_PATH, SYNTHETIC_PATH)
    merger.merge()