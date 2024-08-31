import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

BATCH_SIZE=128
EPOCHS=10
LEARNING_RATE=0.02
# 1 - Download dataset
# 2 - Create data loader
# 3 - Build Model
# 4 - Train
# 5 - Save trained model

class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(28*28,256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.softmax=nn.Softmax(dim=1)
    
    def forward(self,input_data):
        flattened_data = self.flatten(input_data)
        logits=self.dense_layers(flattened_data)
        predictions=self.softmax(logits)
        return predictions


def download_mnist_datasets():
    train_data= datasets.MNIST(
        root="data",
        download=True,
        train=True,
        transform=ToTensor()
    )
    validation_data= datasets.MNIST(
        root="data",
        download=True,
        train=False,
        transform=ToTensor()
    )
    return train_data, validation_data

def train_one_epoch(model,data_loader,loss_fn,optimizer,device):
    for inputs, targets in data_loader:
        inputs, targets=inputs.to(device), targets.to(device)

        # Calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        
        # Backpropagate loss and update weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Loss: {loss.item()}")


def train(model,data_loader,loss_fn,optimizer,device,n_epochs):
    for i in range(n_epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model,data_loader,loss_fn,optimizer,device)
        print("------------------------------")
    print("Training Completed")

if __name__ == "__main__":
    device="cuda" if torch.cuda.is_available() else "cpu"
    train_data,_=download_mnist_datasets()
    #print(f"MNISt Dataset downloaded and train data shape is {train_data}")

    #Create a Dataloader
    train_data_loader=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)

    #Build model
    print(f"Using {device} device for training")
    feed_forward_net=FeedForwardNet().to(device)

    #Instantiate loss functiona on optimizer
    loss_fn=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(feed_forward_net.parameters(),lr=LEARNING_RATE)

    # Train Model
    train(feed_forward_net,train_data_loader,loss_fn,optimizer,device,EPOCHS)

    torch.save(feed_forward_net.state_dict(), "feedforward.pth")
    print("Model trained and stored at feedforward.pth")

