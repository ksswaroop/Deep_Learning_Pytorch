import torch
from torch import nn
from torch.utils.data import DataLoader
from urbansounddataset import UrbanSoundDataset
import torchaudio
from cnn import CNNNetwork

BATCH_SIZE=128
EPOCHS=10
LEARNING_RATE=0.02

if torch.cuda.is_available():
        device = "cuda"
else:
        device = "cpu"
    #print(f"Using the device {device}")
#device="cpu"

ANNOTATIONS_FILE="C:\\Users\\ksais\\Documents\\Coding\\Datasets\\UrbanSound8K\\UrbanSound8K.csv"
AUDIO_DIR="C:\\Users\\ksais\\Documents\\Coding\\Datasets\\UrbanSound8K\\audio"
SAMPLE_RATE=22050
NUM_SAMPLES=22050

def create_data_loader(train_data, batch_size):
    train_dataloader= DataLoader(train_data,batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model,data_loader,loss_fn,optimizer,device):
    for inputs, targets in data_loader:
        inputs, targets=inputs.to(device), targets.to(device)

        # Calculate loss
        predictions = model(inputs).to(device)
        loss = loss_fn(predictions, targets)
        
        # Backpropagate loss and update weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Loss: {loss.item()}")


def train(model,data_loader,loss_fn,optimizer,device,n_epochs):
    for i in range(n_epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model,data_loader,loss_fn,optimizer,device)
        print("------------------------------")
    print("Training Completed")

if __name__ == "__main__":
    device="cuda" if torch.cuda.is_available() else "cpu"
    #Instantiating dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate= SAMPLE_RATE,
        n_fft =1024,
        hop_length=512,
        n_mels=64
    ).to(device)
    #ms = mel_spectrogram(signal)
    usd= UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR,mel_spectrogram,SAMPLE_RATE,
                           NUM_SAMPLES,device)
   
    #Create a Dataloader
    train_dataloader=create_data_loader(usd,batch_size=BATCH_SIZE)#,shuffle=True)

    #Build model
    print(f"Using {device} device for training")
    cnn=CNNNetwork().to(device)

    #Instantiate loss functiona on optimizer
    loss_fn=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(cnn.parameters(),lr=LEARNING_RATE)

    # Train Model
    train(cnn,train_dataloader,loss_fn,optimizer,device,EPOCHS)

    # Save Model
    torch.save(cnn.state_dict(), "cnnnetwork.pth")
    print("Model trained and stored at cnnnetwork.pth")

