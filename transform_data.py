import cv2
import numpy as np
import random
from pathlib import Path
import shutil
import sys

# Try to import tqdm for a progress bar
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not installed. Run 'pip install tqdm' for a progress bar.")


    def tqdm(iterable):
        return iterable


class DatasetSynthesizer:
    def __init__(self, source_dir, bg_dir, output_dir):
        self.source_dir = Path(source_dir)
        self.bg_dir = Path(bg_dir)
        self.output_dir = Path(output_dir)

        # EXACT NAMES FROM YOUR SCREENSHOT
        self.subsets = ['train', 'valid', 'test']
        self.supported_ext = ['.jpg', '.jpeg', '.png', '.bmp']

        self.backgrounds = self._load_backgrounds()

    def _load_backgrounds(self):
        if not self.bg_dir.exists():
            print(f"ERROR: Background folder not found at {self.bg_dir}")
            sys.exit(1)

        bgs = []
        for ext in self.supported_ext:
            bgs.extend(list(self.bg_dir.glob(f'*{ext}')))

        if not bgs:
            print(f"ERROR: No images found in {self.bg_dir}. Please add some JPG/PNG files.")
            sys.exit(1)

        print(f"Loaded {len(bgs)} background images.")
        return [cv2.imread(str(p)) for p in bgs]

    def remove_white_bg(self, image):
        """
        Creates a mask where white pixels are removed.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Sensitivity settings for "White"
        lower_white = np.array([0, 0, 210])
        upper_white = np.array([180, 40, 255])

        mask = cv2.inRange(hsv, lower_white, upper_white)
        return cv2.bitwise_not(mask)

    def process(self):
        print(f"Starting transformation...")
        print(f"Source: {self.source_dir}")
        print(f"Output: {self.output_dir}")

        # Dictionary to store counts for final report
        stats = {'train': 0, 'valid': 0, 'test': 0, 'total': 0}

        for subset in self.subsets:
            img_source_dir = self.source_dir / subset / 'images'
            lbl_source_dir = self.source_dir / subset / 'labels'

            if not img_source_dir.exists():
                print(f"Skipping {subset}: Could not find {img_source_dir}")
                continue

            # Setup output folders
            img_out_dir = self.output_dir / subset / 'images'
            lbl_out_dir = self.output_dir / subset / 'labels'
            img_out_dir.mkdir(parents=True, exist_ok=True)
            lbl_out_dir.mkdir(parents=True, exist_ok=True)

            # Get images
            images = []
            for ext in self.supported_ext:
                images.extend(list(img_source_dir.glob(f'*{ext}')))

            print(f"\nProcessing {subset} ({len(images)} images)...")

            subset_count = 0  # Counter for this folder

            for img_path in tqdm(images):
                # 1. Read Image
                original = cv2.imread(str(img_path))
                if original is None: continue

                # 2. Create Mask
                mask = self.remove_white_bg(original)
                bottle_part = cv2.bitwise_and(original, original, mask=mask)

                # 3. Prepare Background
                bg_img = random.choice(self.backgrounds).copy()
                bg_img = cv2.resize(bg_img, (original.shape[1], original.shape[0]))

                # 4. Merge
                bg_part = cv2.bitwise_and(bg_img, bg_img, mask=cv2.bitwise_not(mask))
                final_img = cv2.add(bg_part, bottle_part)

                # 5. Save Image
                cv2.imwrite(str(img_out_dir / img_path.name), final_img)

                # 6. Copy Label
                label_name = img_path.stem + '.txt'
                src_label = lbl_source_dir / label_name
                if src_label.exists():
                    shutil.copy(src_label, lbl_out_dir / label_name)

                subset_count += 1

            # Update stats
            stats[subset] = subset_count
            stats['total'] += subset_count

        # === FINAL REPORT ===
        print("\n" + "=" * 30)
        print("  SYNTHESIS COMPLETE REPORT")
        print("=" * 30)
        print(f"  Train Images : {stats['train']}")
        print(f"  Valid Images : {stats['valid']}")
        print(f"  Test Images  : {stats['test']}")
        print("-" * 30)
        print(f"  TOTAL SAVED  : {stats['total']}")
        print("=" * 30)
        print(f"New dataset location: {self.output_dir}")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    original_path = '/home/kalgaonp/anaconda3/envs/kdThesisEnv/mixology/Mixology/bottles-1'
    bg_path = 'backgrounds'
    new_path = 'bottles_synthetic'

    syn = DatasetSynthesizer(original_path, bg_path, new_path)
    syn.process()