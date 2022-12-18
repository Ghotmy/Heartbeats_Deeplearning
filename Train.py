import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from HeartDataset import HeartDataset
import pandas as pd
from CNN import CNNNetwork
from FNN import FNNNetwork

BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001
SAMPLE_RATE = 22050
NUM_SAMPLES = 220500

SET_A_CSV = "/home/ghotmy/College/patterns/heart_beat_DeepLearning/heart-beat-dataset/set_a.csv"
SET_B_CSV = "/home/ghotmy/College/patterns/heart_beat_DeepLearning/heart-beat-dataset/set_b.csv"
DATA_DIR = "/home/ghotmy/College/patterns/heart_beat_DeepLearning/heart-beat-dataset"


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input_t, target in data_loader:
        input_t, target = input_t.to(device), target.to(device)
        # print(target)

        # calculate loss
        prediction = model(input_t)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":

    print("Train:\n1-FNN\n2-CNN\nChoose:")
    network_type = int(input())

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    TotalCSV = pd.read_csv("train.csv")
    # TotalCSV = TotalCSV[(TotalCSV.label.notnull()) & (TotalCSV.label != "artifact")]
    # TotalCSV['label'] = TotalCSV['label'].replace(['normal', 'murmur', 'extrahls'], [0, 1, 2])
    # TotalCSV = TotalCSV.sample(frac=1).reset_index(drop=True)
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

    train_dataloader = create_data_loader(h_data, BATCH_SIZE)

    loss_fn = nn.CrossEntropyLoss()

    if network_type == 1:
        fnn = FNNNetwork(1 * 64 * 431, 256*2, 4).to(device)
        optimiser = torch.optim.Adam(fnn.parameters(), lr=LEARNING_RATE)
        train(fnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)
        torch.save(fnn.state_dict(), "fnn.pth")
        print("Trained FNN saved at fnn.pth")
    else:
        cnn = CNNNetwork().to(device)
        optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
        train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)
        torch.save(cnn.state_dict(), "cnn.pth")
        print("Trained CNN saved at cnn.pth")
