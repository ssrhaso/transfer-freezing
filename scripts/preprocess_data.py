import os
import glob
import numpy as np
import librosa
import torch
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def get_log_melspec(audio_path, sr=22050):
    """
    Converts audio to a Log-Mel Spectrogram.
    Explanations:
    - sr=22050: Standard sample rate for music analysis.
    - n_fft=2048: Window size for Fourier Transform (time resolution).
    - hop_length=512: Overlap between windows.
    - n_mels=128: Height of the spectrogram (frequency resolution).
    """
    y, _ = librosa.load(audio_path, sr=sr)
    melspec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128
    )
    return np.log1p(melspec)  # Log-scaling makes weak sounds visible

def resize_spec(spec, size=(224, 224)):
    """
    Resizes the spectrogram to match ResNet/ViT input requirements.
    The models expect 224x224 pixel inputs.
    """
    return cv2.resize(spec, dsize=size, interpolation=cv2.INTER_LINEAR)

def process_split(files, split_name, output_root, genres):
    save_dir = os.path.join(output_root, split_name)
    os.makedirs(save_dir, exist_ok=True)
    
    for f in tqdm(files, desc=f"Processing {split_name}"):
        try:
            genre = os.path.basename(os.path.dirname(f))  # FIX: Use os methods
            label = genres.index(genre)
            
            # Compute & Resize
            spec = get_log_melspec(f)
            spec_resized = resize_spec(spec)
            
            # Normalize to [0, 1] BEFORE saving
            spec_min = spec_resized.min()
            spec_max = spec_resized.max()
            spec_normalized = (spec_resized - spec_min) / (spec_max - spec_min + 1e-8)
            
            # Save as Tensor (now in [0, 1] range like images)
            data = torch.tensor(spec_normalized, dtype=torch.float32)
            fname = f"{genre}_{os.path.basename(f).replace('.wav', '.pt')}"
            
            torch.save({'data': data, 'label': label}, os.path.join(save_dir, fname))
            
        except Exception as e:
            print(f" Error {f}: {e}")


if __name__ == "__main__":
    # PATHS
    RAW_PATH = "./data/gtzan/genres_original"
    OUT_PATH = "./data/processed"
    
    # 1. Find all audio files
    all_files = glob.glob(os.path.join(RAW_PATH, "*", "*.wav"))
    
    # Extract unique genres from folder names
    genres = sorted(list(set([f.split('/')[-2] for f in all_files])))
    labels = [f.split('/')[-2] for f in all_files]
    
    # 2. Split the data (Stratified)
    # We split 80% for training, 20% for temp (val/test)
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        all_files, labels, test_size=0.2, random_state=1, stratify=labels
    )
    
    # Split the temp 20% into 10% validation and 10% test
    val_files, test_files, _, _ = train_test_split(
        temp_files, temp_labels, test_size=0.5, random_state=1, stratify=temp_labels
    )
    
    # 3. Run the processing
    process_split(train_files, 'train', OUT_PATH, genres)
    process_split(val_files, 'val', OUT_PATH, genres)
    process_split(test_files, 'test', OUT_PATH, genres)
    
    print("\n")
    print("PREPROCESSING COMPLETE. TRAINING READY.")
