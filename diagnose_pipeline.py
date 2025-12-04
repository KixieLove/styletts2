#!/usr/bin/env python3
"""
Comprehensive pipeline diagnostic - validates each stage step by step
"""
import os
import sys
import numpy as np
import torch

def test_1_check_data_files():
    """Step 1: Verify audio files exist and are readable"""
    print("\n" + "="*70)
    print("STEP 1: Check Data Files")
    print("="*70)
    
    data_dir = "Data/angelina/wavs24k"
    filelist = "filelists/train.txt"
    
    if not os.path.exists(filelist):
        print(f"❌ Filelist not found: {filelist}")
        return False
    
    with open(filelist, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"✓ Found filelist with {len(lines)} entries")
    
    # Check first 5 files
    import soundfile as sf
    bad_files = []
    for i, line in enumerate(lines[:5]):
        parts = line.strip().split('|')
        wav_path = os.path.join(data_dir, parts[0])
        
        if not os.path.exists(wav_path):
            print(f"❌ File not found: {wav_path}")
            bad_files.append(wav_path)
            continue
        
        try:
            wav, sr = sf.read(wav_path)
            has_nan = np.isnan(wav).any()
            has_inf = np.isinf(wav).any()
            
            status = "✓" if not (has_nan or has_inf) else "❌"
            print(f"{status} {parts[0]}: shape={wav.shape}, sr={sr}, NaN={has_nan}, Inf={has_inf}")
            
            if has_nan or has_inf:
                bad_files.append(wav_path)
        except Exception as e:
            print(f"❌ Error reading {parts[0]}: {e}")
            bad_files.append(wav_path)
    
    if bad_files:
        print(f"\n⚠️  Found {len(bad_files)} problematic files")
        return False
    
    print("\n✓ All sampled files are readable and contain no NaN/Inf")
    return True


def test_2_check_dataset():
    """Step 2: Test FilePathDataset loading"""
    print("\n" + "="*70)
    print("STEP 2: Test FilePathDataset")
    print("="*70)
    
    try:
        from meldataset import FilePathDataset
        
        with open("filelists/train.txt", 'r', encoding='utf-8') as f:
            path_list = f.readlines()
        
        dataset = FilePathDataset(
            data_list=path_list[:10],  # Just first 10
            root_path="Data/angelina",
            validation=False
        )
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        # Try loading first sample
        try:
            sample = dataset[0]
            speaker_id, acoustic_feature, text_tensor, ref_text, ref_mel_tensor, ref_label, path, wave = sample
            
            print(f"✓ Sample 0 loaded:")
            print(f"  - speaker_id: {speaker_id}")
            print(f"  - acoustic_feature (mel): shape={acoustic_feature.shape}, NaN={torch.isnan(acoustic_feature).any().item()}")
            print(f"  - text_tensor: shape={text_tensor.shape}")
            print(f"  - wave (numpy): shape={wave.shape}, dtype={wave.dtype}, NaN={np.isnan(wave).any()}, Inf={np.isinf(wave).any()}")
            print(f"  - path: {path}")
            
            if np.isnan(wave).any() or np.isinf(wave).any():
                print("❌ Wave contains NaN/Inf!")
                return False
            
            print("✓ Sample loaded successfully with no NaN/Inf in waveform")
            return True
            
        except Exception as e:
            print(f"❌ Error loading sample 0: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_check_collater():
    """Step 3: Test Collater (batch creation)"""
    print("\n" + "="*70)
    print("STEP 3: Test Collater (Batch Creation)")
    print("="*70)
    
    try:
        from meldataset import FilePathDataset, Collater
        
        with open("filelists/train.txt", 'r', encoding='utf-8') as f:
            path_list = f.readlines()
        
        dataset = FilePathDataset(
            data_list=path_list[:10],
            root_path="Data/angelina",
            validation=False
        )
        
        collater = Collater(return_wave=False)
        
        # Manually create a batch
        batch_samples = [dataset[i] for i in range(min(4, len(dataset)))]
        
        try:
            batch = collater(batch_samples)
            waves, texts, input_lengths, ref_texts, ref_lengths, mels, output_lengths, ref_mels = batch
            
            print(f"✓ Batch created with {len(waves)} samples")
            print(f"  - mels: shape={mels.shape}, NaN={torch.isnan(mels).any().item()}")
            print(f"  - texts: shape={texts.shape}, NaN={torch.isnan(texts).any().item()}")
            print(f"  - waves list: {len(waves)} items")
            
            for i, w in enumerate(waves):
                if isinstance(w, np.ndarray):
                    has_nan = np.isnan(w).any()
                    has_inf = np.isinf(w).any()
                    status = "✓" if not (has_nan or has_inf) else "❌"
                    print(f"  {status} waves[{i}]: shape={w.shape}, dtype={w.dtype}, NaN={has_nan}, Inf={has_inf}")
                    
                    if has_nan or has_inf:
                        return False
            
            print("✓ All waveforms in batch are valid (no NaN/Inf)")
            return True
            
        except Exception as e:
            print(f"❌ Error creating batch: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Error in collater test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_check_dataloader():
    """Step 4: Test full DataLoader"""
    print("\n" + "="*70)
    print("STEP 4: Test DataLoader")
    print("="*70)
    
    try:
        from meldataset import build_dataloader
        
        dataloader = build_dataloader(
            path_list="filelists/train.txt",
            root_path="Data/angelina",
            batch_size=4,
            num_workers=0,
            validation=False,
            device='cpu'
        )
        
        print(f"✓ DataLoader created")
        
        # Get first batch
        batch = next(iter(dataloader))
        waves, texts, input_lengths, ref_texts, ref_lengths, mels, output_lengths, ref_mels = batch
        
        print(f"✓ First batch loaded:")
        print(f"  - batch_size: {len(waves)}")
        print(f"  - mels: shape={mels.shape}, NaN={torch.isnan(mels).any().item()}")
        print(f"  - texts: shape={texts.shape}, NaN={torch.isnan(texts).any().item()}")
        
        all_valid = True
        for i, w in enumerate(waves):
            if isinstance(w, np.ndarray):
                has_nan = np.isnan(w).any()
                has_inf = np.isinf(w).any()
                status = "✓" if not (has_nan or has_inf) else "❌"
                print(f"  {status} waves[{i}]: NaN={has_nan}, Inf={has_inf}")
                if has_nan or has_inf:
                    all_valid = False
        
        if not all_valid:
            print("❌ Some waveforms contain NaN/Inf")
            return False
        
        print("✓ DataLoader working correctly - no NaN/Inf detected")
        return True
        
    except Exception as e:
        print(f"❌ Error in dataloader test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "█"*70)
    print("COMPREHENSIVE PIPELINE DIAGNOSTIC")
    print("█"*70)
    
    results = {
        "Data Files": test_1_check_data_files(),
        "FilePathDataset": test_2_check_dataset(),
        "Collater": test_3_check_collater(),
        "DataLoader": test_4_check_dataloader(),
    }
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for step, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {step}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓✓✓ All pipeline stages validated successfully! ✓✓✓")
        print("\nYour data pipeline is working correctly.")
        print("The NaN issue must be in the training loop itself.")
        return 0
    else:
        print("\n❌❌❌ Pipeline validation failed ❌❌❌")
        print("\nOne or more stages have issues. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
