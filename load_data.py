import os
import struct
import gzip
import shutil
import numpy as np
import matplotlib.pyplot as plt

# ======================
# Paths
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Available EMNIST datasets
EMNIST_DATASETS = [
    "emnist-balanced",
    "emnist-byclass",
    "emnist-bymerge",
    "emnist-digits",
    "emnist-letters",
    "emnist-mnist",
]


# ======================
# MNIST extraction & loading
# ======================
MNIST_GZ_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def extract_mnist():
    for file in MNIST_GZ_FILES:
        gz_path = os.path.join(DATA_DIR, file)
        out_path = gz_path[:-3]

        if os.path.exists(out_path):
            continue  # already extracted

        with gzip.open(gz_path, "rb") as f_in:
            with open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        print(f"Extracted {file}")


def load_mnist_images(filename):
    """Load MNIST images from file."""
    with open(os.path.join(DATA_DIR, filename), "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        return images.reshape(num, rows, cols)


def load_mnist_labels(filename):
    """Load MNIST labels from file."""
    with open(os.path.join(DATA_DIR, filename), "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_mnist_dataset(split: str = "train"):
    """
    Load MNIST dataset (digits 0-9).
    
    Args:
        split: 'train' or 'test'
    
    Returns:
        Tuple of (images, labels)
    """
    if split == "train":
        images = load_mnist_images("train-images-idx3-ubyte")
        labels = load_mnist_labels("train-labels-idx1-ubyte")
    elif split == "test":
        images = load_mnist_images("t10k-images-idx3-ubyte")
        labels = load_mnist_labels("t10k-labels-idx1-ubyte")
    else:
        raise ValueError("split must be 'train' or 'test'")
    
    return images, labels


# ======================
# EMNIST loading functions
# ======================
def load_emnist_images(dataset_name: str, split: str = "train"):
    """
    Load EMNIST images for a specific dataset and split.
    
    Args:
        dataset_name: One of 'balanced', 'byclass', 'bymerge', 'digits', 'letters', 'mnist'
        split: 'train' or 'test'
    
    Returns:
        numpy array of shape (num_samples, 28, 28)
    """
    filename = f"emnist-{dataset_name}-{split}-images-idx3-ubyte"
    filepath = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        return images.reshape(num, rows, cols)


def load_emnist_labels(dataset_name: str, split: str = "train"):
    """
    Load EMNIST labels for a specific dataset and split.
    
    Args:
        dataset_name: One of 'balanced', 'byclass', 'bymerge', 'digits', 'letters', 'mnist'
        split: 'train' or 'test'
    
    Returns:
        numpy array of labels
    """
    filename = f"emnist-{dataset_name}-{split}-labels-idx1-ubyte"
    filepath = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_emnist_dataset(dataset_name: str, split: str = "train"):
    """
    Load both images and labels for a specific EMNIST dataset.
    
    Args:
        dataset_name: One of 'balanced', 'byclass', 'bymerge', 'digits', 'letters', 'mnist'
        split: 'train' or 'test'
    
    Returns:
        Tuple of (images, labels)
    """
    images = load_emnist_images(dataset_name, split)
    labels = load_emnist_labels(dataset_name, split)
    return images, labels


def list_available_emnist_datasets():
    """List all available EMNIST datasets."""
    print("Available EMNIST datasets:")
    for dataset in EMNIST_DATASETS:
        for split in ["train", "test"]:
            train_img = os.path.join(DATA_DIR, f"emnist-{dataset}-{split}-images-idx3-ubyte")
            if os.path.exists(train_img):
                images = load_emnist_images(dataset, split)
                labels = load_emnist_labels(dataset, split)
                print(f"  {dataset:15s} ({split:5s}): {images.shape[0]:6d} samples")


# ======================
# Debug / test
# ======================
if __name__ == "__main__":
    print("DATA DIR:", DATA_DIR)
    print("FILES:", os.listdir(DATA_DIR))
    print()

    # Example: Load MNIST
    print("=== MNIST Dataset ===")
    try:
        images, labels = load_mnist_dataset("train")
        print(f"MNIST (train): {images.shape}, labels: {labels.shape}")
        plt.figure(figsize=(8, 3))
        plt.imshow(images[0].reshape(28, 28), cmap="gray")
        plt.title(f"Digit: {labels[0]}")
        plt.show()
    except FileNotFoundError as e:
        print(f"Error: {e}")
    print()

    # Example: Load EMNIST datasets
    print("=== EMNIST Datasets ===")
    list_available_emnist_datasets()
    print()

    # Load different EMNIST datasets
    print("=== Loading EMNIST examples ===")
    
    # Load digits dataset
    try:
        images, labels = load_emnist_dataset("digits", "train")
        print(f"Digits (train): {images.shape}, labels: {labels.shape}")
        plt.figure(figsize=(8, 3))
        plt.imshow(images[0].reshape(28, 28), cmap="gray")
        plt.title(f"Digit: {labels[0]}")
        plt.show()
    except FileNotFoundError as e:
        print(f"Error: {e}")
    
    # Load letters dataset
    try:
        images, labels = load_emnist_dataset("letters", "train")
        print(f"Letters (train): {images.shape}, labels: {labels.shape}")
        plt.figure(figsize=(8, 3))
        plt.imshow(images[0].reshape(28, 28), cmap="gray")
        plt.title(f"Letter: {chr(labels[0] + 96)}")  # Convert to letter
        plt.show()
    except FileNotFoundError as e:
        print(f"Error: {e}")
    
    # Load balanced dataset
    try:
        images, labels = load_emnist_dataset("balanced", "train")
        print(f"Balanced (train): {images.shape}, labels: {labels.shape}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
