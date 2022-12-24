import torch
import torchaudio
import pandas as pd
import matplotlib.pyplot as plt

from CNN import CNNNetwork
from FNN import FNNNetwork
from VGG import VGGNetwork
from RNN import RNN
from CNN_ResNet import CNN_ResNet

from HeartDataset import HeartDataset
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from Train import DATA_DIR, SAMPLE_RATE, NUM_SAMPLES, class_mapping


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


if __name__ == "__main__":

    print("Predict:\n1-FNN\n2-CNN\n3-CNN:ResNet\n4-CNN:VGG\n5-RNN:LSTM\nChoose:\n")
    network_type = int(input())
    model = None
    state_dict = None
    if (network_type == 1):
        model = FNNNetwork(1 * 64 * 431, 32, 4)
        state_dict = torch.load("fnn.pth")
    elif network_type == 2:
        model = CNNNetwork()
        state_dict = torch.load("cnn.pth")
    elif network_type == 3:
        model = CNN_ResNet().GetModel()
        state_dict = torch.load("resnet.pth")
    elif network_type == 4:
        model = VGGNetwork()
        state_dict = torch.load("vgg.pth")
    elif network_type == 5:
        model = RNN()
        state_dict = torch.load("rnn.pth")

    print("Use best epoch ?:\n1-Yes\n2-NO!\nChoose:\n")
    best = int(input())
    if best == 1:
        state_dict = torch.load("BestEpoch.pth")

    model.load_state_dict(state_dict)
    TotalCSV = pd.read_csv("test.csv")
    print(TotalCSV.to_string())

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
                          "cpu")

    success = 0
    y_true, y_pred = list(), list()
    print(f'LENGHT={len(h_data)}')
    for i in range(len(h_data)):
        input_t, target = h_data[i][0], h_data[i][1]
        input_t.unsqueeze_(0)
        predicted, expected = predict(model, input_t, target, class_mapping)
        y_true.append(expected)
        y_pred.append(predicted)
        if predicted == expected:
            success += 1
        print(f"Predicted: '{predicted}', expected: '{expected}'->{success}")
    print(f'Success= {success}')
    print(f'Acc= {success / len(h_data)}')
    print('=' * 65)
    print(classification_report(y_true, y_pred, zero_division=True))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["extrahls", "extrastole", "murmur", "normal"])
    disp.plot()
    plt.show()
