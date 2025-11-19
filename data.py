import yaml
from pathlib import Path


class DatasetManager:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.data_yaml = self.dataset_path / 'data.yaml'

    def load_config(self):
        """Load the data.yaml configuration"""
        with open(self.data_yaml, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def verify_dataset(self):
        """Verify dataset structure and print statistics"""
        config = self.load_config()

        print("Dataset Configuration:")
        print(f"Classes: {config['names']}")
        print(f"Number of classes: {config['nc']}")

        # Count images in each split
        train_path = self.dataset_path / 'train' / 'images'
        valid_path = self.dataset_path / 'valid' / 'images'
        test_path = self.dataset_path / 'test' / 'images'

        print(f"\nDataset Statistics:")
        print(f"Training images: {len(list(train_path.glob('*')))} if train_path.exists() else 0")
        print(f"Validation images: {len(list(valid_path.glob('*')))} if valid_path.exists() else 0")
        print(f"Test images: {len(list(test_path.glob('*')))} if test_path.exists() else 0")

        return config


if __name__ == "__main__":
    # Test the dataset manager
    dm = DatasetManager('./data/your-dataset-name')
    dm.verify_dataset()