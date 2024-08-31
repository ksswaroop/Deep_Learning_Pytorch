import torch
import torchaudio

from train import FeedForwardNet, download_mnist_datasets

class_mapping= [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
    ]

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        #Tensor (1 #No of samples, 10 #No of classes) -> [[0.1,0.01,......,0.6]]
        predicted_index=predictions[0].argmax()
        predicted = class_mapping[predicted_index]
        expected= class_mapping[target]
    return predicted, expected


if __name__ == "__main__":
    # Load back the model
    feed_forward_net=FeedForwardNet()
    state_dict=torch.load("C:\\Users\\ksais\\Documents\\Coding\\Pytorch practice\\Youtube Audio\\03 make Predictions\\feedforward.pth")
    feed_forward_net.load_state_dict(state_dict)

    # Load MNIST validation dataset
    _,validation_data=download_mnist_datasets()
    # Get sample from validation dataset for inference
    input,target = validation_data[0][0], validation_data[0][1]
    # Make an inference
    predicted, expected = predict(feed_forward_net,input,target,class_mapping)
    print(f"Predcited: {predicted} , Expected : {expected}")