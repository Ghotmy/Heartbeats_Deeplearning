import torchaudio
from HeartDataset import HeartDataset
import pandas as pd
import numpy as np
from Train import DATA_DIR, SAMPLE_RATE, NUM_SAMPLES

SET_A_CSV = "/home/ghotmy/College/patterns/heart_beat_DeepLearning/heart-beat-dataset/set_a.csv"
SET_B_CSV = "/home/ghotmy/College/patterns/heart_beat_DeepLearning/heart-beat-dataset/set_b.csv"

##Combine CSV files:
TotalCSV = pd.concat([pd.read_csv(SET_A_CSV), pd.read_csv(SET_B_CSV)], ignore_index=True)
normal = TotalCSV[(TotalCSV['label'] == 'normal')]
murmur = TotalCSV[(TotalCSV['label'] == 'murmur')]
extrahls = TotalCSV[(TotalCSV['label'] == 'extrahls')]
extrastole = TotalCSV[(TotalCSV['label'] == 'extrastole')]

TotalDataBalanced = normal  # .sample(300)
TotalDataBalanced = TotalDataBalanced.append([murmur] * 2, ignore_index=True)
TotalDataBalanced = TotalDataBalanced.append([extrahls] * 10, ignore_index=True)
TotalDataBalanced = TotalDataBalanced.append([extrastole] * 4, ignore_index=True)
print(TotalDataBalanced.to_string())

TotalDataBalanced['label'] = TotalDataBalanced['label'].replace(["normal", "murmur", "extrahls", "extrastole"],
                                                                [0, 1, 2, 3])
train, validation, test = np.split(TotalDataBalanced.sample(frac=1),
                                   [int(0.70 * len(TotalDataBalanced)),
                                    int(0.85 * len(TotalDataBalanced))])  # Split 75 15 15

print(TotalDataBalanced["label"].value_counts())
print(len(train))
print(len(validation))
print(len(test))
# print(train.to_string())

## Save CSVs
train.to_csv('train.csv', index=False)
validation.to_csv('validation.csv', index=False)
test.to_csv('test.csv', index=False)

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

h_data = HeartDataset(train,
                      DATA_DIR,
                      mel_spectrogram,
                      SAMPLE_RATE,
                      NUM_SAMPLES,
                      "cpu")
print(f"There are {len(h_data)} samples in the dataset.")
for i in h_data:
    signal, label = i
    # print(signal.shape)
