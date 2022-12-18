import torch
import torchaudio
from HeartDataset import HeartDataset
import pandas as pd

if __name__ == "__main__":
    SET_A_CSV = "/home/ghotmy/College/patterns/heart_beat_DeepLearning/heart-beat-dataset/set_a.csv"
    SET_B_CSV = "/home/ghotmy/College/patterns/heart_beat_DeepLearning/heart-beat-dataset/set_b.csv"
    DATA_DIR = "/home/ghotmy/College/patterns/heart_beat_DeepLearning/heart-beat-dataset"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 220500

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    ##Combine CSV files:
    # TotalCSV = pd.read_csv(SET_A_CSV)
    TotalCSV = pd.concat([pd.read_csv(SET_A_CSV), pd.read_csv(SET_B_CSV)], ignore_index=True)
    ###Drop unlabeled and artifact and noisy data:
    TotalCSV = TotalCSV[(TotalCSV.label.notnull()) & (TotalCSV.label != "artifact") & TotalCSV.sublabel.isnull()]
    TotalCSV['label'] = TotalCSV['label'].replace(["normal", "murmur", "extrahls", "extrastole"], [0, 1, 2, 3])
    ###Random mix the data:
    TotalCSV = TotalCSV.sample(frac=1).reset_index(drop=True)

    print(TotalCSV.to_string())

    TrainCSV = TotalCSV.iloc[:295, :]
    TestCSV = TotalCSV.iloc[295:, :]

    ###Generate new CSVs
    ##divide data into TEST and Training
    # TotalCSV.to_csv('set_ab.csv', index=False)
    TrainCSV.to_csv('train.csv', index=False)
    TestCSV.to_csv('test.csv', index=False)

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
    print(f"There are {len(h_data)} samples in the dataset.")
    signal, label = h_data[0]
    # print(signal.shape)
    # print(label)
