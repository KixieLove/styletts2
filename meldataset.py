#coding: utf-8
import os
import os.path as osp
import time
import random
import numpy as np
import random
import soundfile as sf
import librosa
import unicodedata

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import pandas as pd


from text_utils import symbols  # import the shared vocab
dicts = {s: i for i, s in enumerate(symbols)}

from text_utils import symbols, _pad  # symbols only if you actually use it below

import unicodedata, re
from text_utils import symbols, _pad  # keep your existing exports

dicts = {s: i for i, s in enumerate(symbols)}
TEXT_VOCAB_LIMIT = 178  # must match ASR embedding size

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
        self.pad_id = dicts.get(_pad, 0)

        # Normalize straight apostrophe to curly (low-id punctuation)
        self._premap = str.maketrans({
            "'": "’",         # ASCII apostrophe -> curly
            "\u00A0": " ",    # NBSP -> space
        })

    def _normalize(self, s: str) -> str:
        s = unicodedata.normalize('NFKC', s)
        s = s.replace('\u2011', '-')  # non-breaking hyphen
        s = s.replace('\u2013', '-')  # en dash
        return s.translate(self._premap)

    def __call__(self, text: str):
        text = self._normalize(text)
        out = []
        for ch in text:
            # try direct
            idx = self.word_index_dictionary.get(ch, None)

            # fallback: strip diacritics (á->a) if direct failed
            if idx is None:
                base = ''.join(c for c in unicodedata.normalize('NFKD', ch)
                               if unicodedata.category(c) != 'Mn')
                idx = self.word_index_dictionary.get(base, None)

            # final guard: drop anything that would exceed ASR vocab
            if idx is not None and idx < TEXT_VOCAB_LIMIT:
                out.append(idx)
            # else: silently skip
        return out

np.random.seed(1)
random.seed(1)
SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}
MEL_PARAMS = {
    "n_mels": 80,
}

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    
    # Check for NaN/Inf in input wave
    if torch.isnan(wave_tensor).any() or torch.isinf(wave_tensor).any():
        print(f"[WARNING] Input wave contains NaN/Inf, returning zeros")
        return torch.zeros(1, 80, 192)  # return dummy mel shape
    
    mel_tensor = to_mel(wave_tensor)
    
    # Clamp mel to avoid log of very small numbers
    mel_tensor = torch.clamp(mel_tensor, min=1e-5)
    
    mel_tensor = (torch.log(mel_tensor.unsqueeze(0)) - mean) / std
    
    # Final check and sanitization
    if torch.isnan(mel_tensor).any():
        print(f"[WARNING] Mel preprocessing produced NaN, clamping to [-10, 10]")
        mel_tensor = torch.clamp(mel_tensor, min=-10.0, max=10.0)
    
    if torch.isinf(mel_tensor).any():
        print(f"[WARNING] Mel preprocessing produced Inf, replacing with silence")
        mel_tensor = torch.zeros_like(mel_tensor) - 10.0  # silence value
    
    return mel_tensor

class FilePathDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 root_path,
                 sr=24000,
                 data_augmentation=False,
                 validation=False,
                 OOD_data="Data/OOD_texts.txt",
                 min_length=50,
                 ):

        spect_params = SPECT_PARAMS
        mel_params = MEL_PARAMS

        _data_list = [l.strip().split('|') for l in data_list]
        self.data_list = [data if len(data) == 3 else (*data, 0) for data in _data_list]
        self.text_cleaner = TextCleaner()
        self.sr = sr

        self.df = pd.DataFrame(self.data_list)

        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = -4, 4
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192
        
        self.min_length = min_length
        with open(OOD_data, 'r', encoding='utf-8') as f:
            tl = f.readlines()
        idx = 1 if '.wav' in tl[0].split('|')[0] else 0
        self.ptexts = [t.split('|')[idx] for t in tl]
        
        self.root_path = root_path

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Retry mechanism: if a sample fails to load, try other samples
        max_retries = 5
        for attempt in range(max_retries):
            try:
                data = self.data_list[idx]
                path = data[0]
                
                wave, text_tensor, speaker_id = self._load_tensor(data)
                
                mel_tensor = preprocess(wave).squeeze()
                
                acoustic_feature = mel_tensor.squeeze()
                length_feature = acoustic_feature.size(1)
                acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]
                
                # get reference sample
                ref_data = (self.df[self.df[2] == str(speaker_id)]).sample(n=1).iloc[0].tolist()
                ref_mel_tensor, ref_label = self._load_data(ref_data[:3])
                
                # get OOD text
                ps = ""
                while len(ps) < self.min_length:
                    rand_idx = np.random.randint(0, len(self.ptexts) - 1)
                    ps = self.ptexts[rand_idx]
                    
                    text = self.text_cleaner(ps)
                    text.insert(0, 0)
                    text.append(0)

                    ref_text = torch.LongTensor(text)
                
                return speaker_id, acoustic_feature, text_tensor, ref_text, ref_mel_tensor, ref_label, path, wave
            
            except Exception as e:
                print(f"[WARNING] Failed to load sample at idx {idx} (attempt {attempt + 1}/{max_retries}): {e}")
                # Try a random sample instead
                idx = np.random.randint(0, len(self.data_list))
                if attempt == max_retries - 1:
                    # If all retries fail, return a dummy safe sample
                    print(f"[ERROR] All retries failed, returning dummy sample")
                    dummy_wave = np.random.randn(24000 * 2).astype(np.float32) * 0.01  # 2 sec silence
                    dummy_mel = torch.zeros(80, 192)
                    dummy_text = torch.LongTensor([0, 0])
                    return 0, dummy_mel, dummy_text, dummy_text, dummy_mel, 0, "dummy", dummy_wave

    def _load_tensor(self, data):
        wave_path, text, speaker_id = data
        speaker_id = int(speaker_id)
        
        try:
            wave, sr = sf.read(osp.join(self.root_path, wave_path))
        except Exception as e:
            print(f"[ERROR] Failed to load audio file: {wave_path}, error: {e}")
            # Raise exception to skip this sample in the dataset
            raise RuntimeError(f"Cannot load {wave_path}: {e}")
        
        # Check for NaN/Inf in loaded wave
        if np.isnan(wave).any() or np.isinf(wave).any():
            nan_count = np.isnan(wave).sum()
            inf_count = np.isinf(wave).sum()
            print(f"[CRITICAL] Audio file {wave_path} contains NaN/Inf!")
            print(f"  NaN: {nan_count}/{len(wave)}, Inf: {inf_count}/{len(wave)}, dtype: {wave.dtype}")
            raise RuntimeError(f"Audio file {wave_path} contains NaN/Inf values")
        
        # Check if audio is (nearly) all zeros/silent
        if np.abs(wave).max() < 1e-4:
            print(f"[WARNING] Audio file {wave_path} is silent (max amplitude < 1e-4), adding noise floor")
            wave = wave + np.random.randn(len(wave)) * 1e-5
        
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        if sr != 24000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
            print(wave_path, sr)
            
        wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)
        
        text = self.text_cleaner(text)
        
        text.insert(0, 0)
        text.append(0)
        
        text = torch.LongTensor(text)

        return wave, text, speaker_id

    def _load_data(self, data):
        wave, text_tensor, speaker_id = self._load_tensor(data)
        mel_tensor = preprocess(wave).squeeze()

        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]

        return mel_tensor, speaker_id


class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.min_mel_length = 192
        self.max_mel_length = 192
        self.return_wave = return_wave
        

    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])
        max_rtext_length = max([b[3].shape[0] for b in batch])

        labels = torch.zeros((batch_size)).long()
        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        ref_texts = torch.zeros((batch_size, max_rtext_length)).long()

        input_lengths = torch.zeros(batch_size).long()
        ref_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref_labels = torch.zeros((batch_size)).long()
        paths = ['' for _ in range(batch_size)]
        waves = [None for _ in range(batch_size)]
        
        for bid, (label, mel, text, ref_text, ref_mel, ref_label, path, wave) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            rtext_size = ref_text.size(0)
            labels[bid] = label
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            ref_texts[bid, :rtext_size] = ref_text
            input_lengths[bid] = text_size
            ref_lengths[bid] = rtext_size
            output_lengths[bid] = mel_size
            paths[bid] = path
            ref_mel_size = ref_mel.size(1)
            ref_mels[bid, :, :ref_mel_size] = ref_mel
            
            ref_labels[bid] = ref_label
            waves[bid] = wave

        # --- BEGIN OOV guard ---
        from text_utils import symbols as _SYMBOLS
        _VOCAB_SIZE = len(_SYMBOLS)

        def _bad_ids(t):
            # per-sample max to locate offenders
            return (t >= _VOCAB_SIZE).any().item()

        offenders = []
        for i in range(batch_size):
            if _bad_ids(texts[i]) or _bad_ids(ref_texts[i]):
                offenders.append((i, int(texts[i].max().item()), int(ref_texts[i].max().item()), paths[i]))

        if offenders:
            details = "\n".join([f"  idx={i} max_text={mx_t} max_ref={mx_r} path={p}"
                                for (i, mx_t, mx_r, p) in offenders])
            raise RuntimeError(
                f"OOV token id detected (>= {_VOCAB_SIZE}). Fix transcripts or SYMBOLS.\n{details}"
            )
        # --- END OOV guard ---

        return waves, texts, input_lengths, ref_texts, ref_lengths, mels, output_lengths, ref_mels



def build_dataloader(path_list,
                     root_path,
                     validation=False,
                     OOD_data="Data/OOD_texts.txt",
                     min_length=50,
                     batch_size=8,
                     num_workers=4,
                     device='cpu',
                     collate_config={},
                     dataset_config={}):
    dataset = FilePathDataset(path_list, root_path,
                              OOD_data=OOD_data,
                              min_length=min_length,
                              validation=validation,
                              **dataset_config)
    collate_fn = Collater(**collate_config)

    is_cuda = (device != 'cpu')
    kwargs = dict(
        batch_size=batch_size,
        shuffle=(not validation),
        num_workers=num_workers,
        drop_last=(not validation),
        collate_fn=collate_fn,
        pin_memory=is_cuda,
        persistent_workers=(num_workers > 0),
    )
    # Only pass prefetch_factor when workers > 0 (it’s ignored/invalid otherwise)
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2  # try 2–4

    data_loader = DataLoader(dataset, **kwargs)
    return data_loader
