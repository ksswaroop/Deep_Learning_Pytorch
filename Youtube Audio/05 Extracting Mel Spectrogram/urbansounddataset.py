from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os
import torch

class UrbanSoundDataset(Dataset):
    def __init__(self,annotation_file, audio_dir,transformation,target_sample_rate):
        self.annotations=pd.read_csv(annotation_file)
        self.audio_dir= audio_dir
        self.transformation=transformation
        self.target_sample_rate=target_sample_rate

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal,sr=torchaudio.load(audio_sample_path)
        signal = self._resample_if_necessary(signal,sr)
        # Signal -> (num_channels,samples) -> (2,16000) -> (1,16000)
        signal = self._mix_don_if_necessary(signal)
        signal = self.transformation(signal)
        return signal,label
    
    def _resample_if_necessary(self,signal,sr):
        if sr != self.target_sample_rate:
            resampler=torchaudio.transforms.Resample(sr,self.target_sample_rate)
            signal = resampler(signal)
        return signal
    
    def _mix_donw_if_necessary(self,signal):
        if signal.shape[0]>1: #Get the shape of signal (2,100) -> (1,100)
            signal= torch.mean(signal,dim=0, keepdim=True)
        return signal
        

    
    def _get_audio_sample_path(self,index):
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir,fold,self.annotations.iloc[index,0])
        return path
    def _get_audio_sample_label(self,index):
        return self.annotations.iloc[index,6]


if __name__=="__main__":
    ANNOTATIONS_FILE="C:\\Users\\ksais\\Documents\\Coding\\Datasets\\UrbanSound8K\\UrbanSound8K.csv"
    AUDIO_DIR="C:\\Users\\ksais\\Documents\\Coding\\Datasets\\UrbanSound8K\\audio"
    SAMPLE_RATE=16000
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate= SAMPLE_RATE,
        n_fft =1024,
        hop_length=512,
        n_mels=64
    )
    #ms = mel_spectrogram(signal)
    usd= UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR,mel_spectrogram,SAMPLE_RATE)
    print(f"There are {len(usd)} samples in the dataset.")

    signal,label=usd[0]
    