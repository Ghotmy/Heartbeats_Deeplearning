import torch
import torchaudio
import pandas as pd

from torch import nn
from torch.utils.data import DataLoader
from HeartDataset import HeartDataset

from CNN import CNNNetwork
from FNN import FNNNetwork
from CNN_ResNet import CNN_ResNet

DATA_DIR = "/home/ghotmy/College/patterns/heart_beat_DeepLearning/heart-beat-dataset"
BATCH_SIZE = 200
EPOCHS = 100
LEARNING_RATE = 0.001
SAMPLE_RATE = 22050
NUM_SAMPLES = 110250
best_accuracy = 0

class_mapping = [
    "normal",
    "murmur",
    "extrahls",
    "extrastole"
]


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def calculate_accuracy(predicted, true_labels):
    accuracy = 0
    for i in range(predicted.shape[0]):
        if predicted[i] == true_labels[i]:
            accuracy += 1
    return accuracy


def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        print(f'Preditions:{predictions}')
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


def train_single_epoch(model, train_data_loader, validation_data_loader, loss_fn, optimiser, device):
    # Training model
    model.train()
    loss = 0
    global best_accuracy
    for input_t, target in train_data_loader:
        input_t, target = input_t.to(device), target.to(device)

        # calculate loss
        prediction = model(input_t)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    print(f"Train loss: {loss.item()}")

    # Testing model on validation data
    model.eval()
    total_accuracy = 0
    for input_t, target in validation_data_loader:
        with torch.no_grad():
            input_t, target = input_t.to(device), target.to(device)
            predictions = model(input_t)
            expected = torch.argmax(predictions, dim=1)
            # print(f'EXPECTED:{expected}\nTARGET  :{target}')

            total_accuracy += calculate_accuracy(expected, target)
            loss = loss_fn(predictions, target)
    total_accuracy /= len(val_data)
    print(f"Validation loss: {loss.item()}")
    print(f'Validation Accuracy : {total_accuracy*100}%')
    if total_accuracy > best_accuracy:
        torch.save(model.state_dict(), 'BestEpoch.pth')
        print('\033[91m'+'New Weights are Saved !! '*3 + '\033[0m')
        best_accuracy = total_accuracy


def train(model, train_data_loader, validation_data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_single_epoch(model, train_data_loader, validation_data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":

    print("Train:\n1-FNN\n2-CNN\n3-CNN:ResNet\nChoose:")
    network_type = int(input())

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    TotalCSV = pd.read_csv("train.csv")
    ValidateCSV = pd.read_csv("validation.csv")
    print(TotalCSV.to_string())

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    h_data = HeartDataset(TotalCSV,
                          DATA_DIR,
                          mel_spectrogram,
                          SAMPLE_RATE,
                          NUM_SAMPLES,
                          device)

    val_data = HeartDataset(ValidateCSV,
                            DATA_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)

    train_dataloader = create_data_loader(h_data, BATCH_SIZE)
    validation_dataloader = create_data_loader(val_data, BATCH_SIZE)

    loss_fn = nn.CrossEntropyLoss()

    if network_type == 1:
        fnn = FNNNetwork(1 * 64 * 431, 16, 4).to(device)
        optimiser = torch.optim.Adam(fnn.parameters(), lr=LEARNING_RATE)
        train(fnn, train_dataloader, validation_dataloader, loss_fn, optimiser, device, EPOCHS)
        torch.save(fnn.state_dict(), "fnn.pth")
        print("Trained FNN saved at fnn.pth")
    elif network_type == 2:
        cnn = CNNNetwork().to(device)
        optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
        train(cnn, train_dataloader, validation_dataloader, loss_fn, optimiser, device, EPOCHS)
        torch.save(cnn.state_dict(), "cnn.pth")
        print("Trained CNN saved at cnn.pth")
    else:
        resnet = CNN_ResNet().GetModel().to(device)
        optimiser = torch.optim.Adam(resnet.parameters(), lr=LEARNING_RATE)
        train(resnet, train_dataloader, validation_dataloader, loss_fn, optimiser, device, EPOCHS)
        torch.save(resnet.state_dict(), "resnet.pth")
        print("Trained CNN_ResNet saved at resnet.pth")
