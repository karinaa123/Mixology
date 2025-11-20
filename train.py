from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from data import DatasetManager


class BottleDetectionTrainer:
    def __init__(self, data_yaml_path, model_size='n'):
        """
        Initialize trainer
        model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
        """
        self.data_yaml = data_yaml_path
        self.model = YOLO(f'yolov8{model_size}.pt')
        self.results = None

    def train(self, epochs=50, imgsz=640, batch=16, name='bottle_detector'):
        """Train the model"""
        print(f"Starting training for {epochs} epochs...")

        self.results = self.model.train(
            data=self.data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name=name,
            plots=True,  # Generate plots automatically
            save=True,  # Save checkpoints
            verbose=True
        )

        print(f"\nTraining complete! Model saved to: runs/detect/{name}/")
        return self.results

    def plot_training_metrics(self, save_path='training_metrics.png'):
        """Plot loss curves and metrics"""
        # Load results from CSV
        results_csv = Path(f'runs/detect/{self.results.save_dir.name}/results.csv')

        if not results_csv.exists():
            print("Results CSV not found!")
            return

        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()  # Remove whitespace from column names

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics', fontsize=16)

        # Plot 1: Box Loss
        axes[0, 0].plot(df['train/box_loss'], label='Train Box Loss', color='blue')
        axes[0, 0].plot(df['val/box_loss'], label='Val Box Loss', color='orange')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Box Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Plot 2: Class Loss
        axes[0, 1].plot(df['train/cls_loss'], label='Train Class Loss', color='blue')
        axes[0, 1].plot(df['val/cls_loss'], label='Val Class Loss', color='orange')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Classification Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Plot 3: mAP scores
        axes[1, 0].plot(df['metrics/mAP50(B)'], label='mAP@50', color='green')
        axes[1, 0].plot(df['metrics/mAP50-95(B)'], label='mAP@50-95', color='red')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mAP')
        axes[1, 0].set_title('Mean Average Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Plot 4: Precision and Recall
        axes[1, 1].plot(df['metrics/precision(B)'], label='Precision', color='purple')
        axes[1, 1].plot(df['metrics/recall(B)'], label='Recall', color='brown')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Precision & Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training metrics plot saved to: {save_path}")
        plt.show()

    def evaluate(self):
        """Evaluate the trained model"""
        metrics = self.model.val()
        return metrics


def main():
    # Setup
    DATASET_PATH = 'home/anaconda3/envs/kdThesisEnv/mixology/Mixology/bottles-1'
    DATA_YAML = f'{DATASET_PATH}/data.yaml'

    # Verify dataset
    print("=== Verifying Dataset ===")
    dm = DatasetManager(DATASET_PATH)
    dm.verify_dataset()

    # Train model
    print("\n=== Training Model ===")
    trainer = BottleDetectionTrainer(
        data_yaml_path=DATA_YAML,
        model_size='n'  # Start with nano for faster training
    )

    trainer.train(
        epochs=50,
        imgsz=640,
        batch=16,
        name='bottle_detector_v1'
    )

    # Plot metrics
    print("\n=== Generating Plots ===")
    trainer.plot_training_metrics('training_results.png')

    # Evaluate
    print("\n=== Evaluating Model ===")
    metrics = trainer.evaluate()
    print(f"Final mAP50: {metrics.box.map50:.3f}")
    print(f"Final mAP50-95: {metrics.box.map:.3f}")


if __name__ == "__main__":
    main()