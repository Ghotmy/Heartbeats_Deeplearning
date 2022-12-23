import torchaudio
from HeartDataset import HeartDataset
import pandas as pd
import numpy as np
from Train import DATA_DIR, SAMPLE_RATE, NUM_SAMPLES

SET_A_CSV = "/home/ghotmy/College/patterns/heart_beat_DeepLearning/heart-beat-dataset/set_a.csv"
SET_B_CSV = "/home/ghotmy/College/patterns/heart_beat_DeepLearning/heart-beat-dataset/set_b.csv"

##Combine CSV files:
TotalCSV = pd.concat([pd.read_csv(SET_A_CSV), pd.read_csv(SET_B_CSV)], ignore_index=True)
TotalCSV = TotalCSV[(TotalCSV.label.notnull()) & (TotalCSV.label != "artifact")]
TotalCSV['label'] = TotalCSV['label'].replace(["normal", "murmur", "extrahls", "extrastole"],
                                              [0, 1, 2, 3])

train, validation, test = np.split(TotalCSV.sample(frac=1, random_state=1),
                                   [int(0.70 * len(TotalCSV)),
                                    int(0.85 * len(TotalCSV))])  # Split 75 15 15

print(len(train))
print(len(validation))
print(len(test))
print(train["label"].value_counts())
print(validation["label"].value_counts())
print(test["label"].value_counts())

normal = train[(train['label'] == 0)]
murmur = train[(train['label'] == 1)]
extrahls = train[(train['label'] == 2)]
extrastole = train[(train['label'] == 3)]

train_balanced = normal  # .sample(300)
train_balanced = train_balanced.append([murmur] * 2, ignore_index=True)
train_balanced = train_balanced.append([extrahls] * 10, ignore_index=True)
train_balanced = train_balanced.append([extrastole] * 4, ignore_index=True)
train_balanced = train_balanced.sample(frac=1, random_state=1).reset_index(drop=True)

print(len(train_balanced))
print(train_balanced["label"].value_counts())
print(train_balanced.to_string())

## Save CSVs
train_balanced.to_csv('train.csv', index=False)
validation.to_csv('validation.csv', index=False)
test.to_csv('test.csv', index=False)

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

h_data = HeartDataset(train_balanced,
                      DATA_DIR,
                      mel_spectrogram,
                      SAMPLE_RATE,
                      NUM_SAMPLES,
                      "cpu")
print(f"There are {len(h_data)} samples in the dataset.")
for i in h_data:
    signal, label = i
    print(signal.shape)
