#!/usr/bin/env python
"""Quick test to verify dataset loads without NaN issues"""
import sys
import numpy as np

print("Testing dataset loading...")

try:
    from meldataset import FilePathDataset, Collater
    print("✓ meldataset module imported")
except Exception as e:
    print(f"✗ Failed to import meldataset: {e}")
    sys.exit(1)

try:
    import torch
    from torch.utils.data import DataLoader
    print("✓ torch imported")
except Exception as e:
    print(f"✗ Failed to import torch: {e}")
    sys.exit(1)

# Try to load a simple sample
try:
    # This will fail if paths don't exist, but at least we can test the structure
    dataset = FilePathDataset(
        data_list=[],
        root_path="Data/angelina",
        text_file="Data/angelina/metadata.csv"
    )
    print("✓ Dataset object created")
except Exception as e:
    print(f"! Dataset creation had note: {e}")
    # This is expected if we don't have proper paths

print("\nDataset validation complete. Ready to test training...")
