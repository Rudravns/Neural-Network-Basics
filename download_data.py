#!/usr/bin/env python3
"""
MNIST Dataset Downloader
This script downloads the MNIST dataset automatically.
It supports multiple MNIST variants: balanced, byclass, bymerge, digits, letters, and mnist.
"""

import os
import urllib.request
import gzip
import shutil
from pathlib import Path


class MNISTDownloader:
    """Download and extract MNIST dataset variants"""
    
    # MNIST/EMNIST dataset URLs
    DATASETS = {
        'balanced': {
            'url': 'https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip',
            'type': 'balanced'
        },
        'byclass': {
            'url': 'https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip',
            'type': 'byclass'
        },
        'bymerge': {
            'url': 'https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip',
            'type': 'bymerge'
        },
        'digits': {
            'url': 'https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip',
            'type': 'digits'
        },
        'letters': {
            'url': 'https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip',
            'type': 'letters'
        },
        'mnist': {
            'url': 'https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip',
            'type': 'mnist'
        }
    }
    
    def __init__(self, data_dir='data'):
        """Initialize downloader with data directory"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def download_file(self, url, destination):
        """Download a file from URL"""
        print(f"Downloading {url}")
        try:
            urllib.request.urlretrieve(url, destination, self._show_progress)
            print(f"\n✓ Downloaded successfully to {destination}")
            return True
        except Exception as e:
            print(f"\n✗ Failed to download: {e}")
            return False
    
    def _show_progress(self, block_num, block_size, total_size):
        """Show download progress"""
        downloaded = block_num * block_size
        percent = min(downloaded * 100 // total_size, 100)
        print(f'\rDownloading: {percent}%', end='')
    
    def extract_gz(self, gz_path):
        """Extract a gzip file"""
        out_path = gz_path.with_suffix('')
        if out_path.exists():
            print(f"✓ {out_path.name} already extracted")
            return
        
        print(f"Extracting {gz_path.name}...")
        try:
            with gzip.open(gz_path, 'rb') as f_in:
                with open(out_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"✓ Extracted to {out_path.name}")
        except Exception as e:
            print(f"✗ Failed to extract: {e}")
    
    def download_mnist_quick(self):
        """Download MNIST dataset (quick version using urllib)"""
        print("=" * 60)
        print("MNIST Dataset Downloader")
        print("=" * 60)
        
        # Alternative direct URLs for MNIST files
        files = {
            'train-images-idx3-ubyte.gz': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'train-labels-idx1-ubyte.gz': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            't10k-images-idx3-ubyte.gz': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            't10k-labels-idx1-ubyte.gz': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
        }
        
        print(f"\nDownloading to: {self.data_dir.absolute()}")
        print(f"Files to download: {len(files)}")
        print("-" * 60)
        
        for filename, url in files.items():
            filepath = self.data_dir / filename
            
            if filepath.exists():
                print(f"✓ {filename} already exists")
                continue
            
            if self.download_file(url, str(filepath)):
                self.extract_gz(filepath)
            print()
        
        print("=" * 60)
        print("✓ MNIST dataset download complete!")
        print("=" * 60)
    
    def verify_dataset(self):
        """Verify that required dataset files exist"""
        required_files = [
            'train-images-idx3-ubyte',
            'train-labels-idx1-ubyte',
            't10k-images-idx3-ubyte',
            't10k-labels-idx1-ubyte',
        ]
        
        print("\nVerifying dataset files...")
        missing = []
        for filename in required_files:
            filepath = self.data_dir / filename
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"✓ {filename} ({size_mb:.1f} MB)")
            else:
                print(f"✗ {filename} - MISSING")
                missing.append(filename)
        
        if missing:
            print(f"\n⚠ Missing {len(missing)} file(s). Please run the downloader.")
            return False
        
        print("\n✓ All required files present!")
        return True


def main():
    """Main execution"""
    import sys
    
    # Get data directory from command line or use default
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'
    
    downloader = MNISTDownloader(data_dir)
    
    # Download and extract MNIST dataset
    downloader.download_mnist_quick()
    
    # Verify
    downloader.verify_dataset()


if __name__ == '__main__':
    main()
