import torch
import torchaudio
import pandas as pd

from CNN import CNNNetwork
from HeartDataset import HeartDataset
from Train import DATA_DIR, SET_A_CSV, SET_B_CSV, SAMPLE_RATE, NUM_SAMPLES

class_mapping = [
    "normal",
    "murmur",
    "extrahls"
]


def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":
    # load back the model
    cnn = CNNNetwork()
    state_dict = torch.load("cnn.pth")
    cnn.load_state_dict(state_dict)

    TotalCSV = pd.read_csv("test.csv")
    # TotalCSV = TotalCSV[(TotalCSV.label.notnull()) & (TotalCSV.label != "artifact")]
    # TotalCSV['label'] = TotalCSV['label'].replace(['normal', 'murmur', 'extrahls'], [0, 1, 2])
    TotalCSV = TotalCSV.sample(frac=1).reset_index(drop=True)
    print(TotalCSV.to_string())

    # load urban sound dataset dataset
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

    # get a sample from the urban sound dataset for inference
    success = 0
    for i in range(len(h_data)):
        input_t, target = h_data[i][0], h_data[i][1]  # [batch size, num_channels, fr, time]
        input_t.unsqueeze_(0)
        # make an inference
        predicted, expected = predict(cnn, input_t, target, class_mapping)
        print(f"Predicted: '{predicted}', expected: '{expected}'")
        if predicted == expected:
            success += 1
    print(f'Success= {success}')
    print(f'Acc= {success/len(h_data)}')