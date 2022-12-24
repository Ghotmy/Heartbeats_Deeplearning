from torch import nn
from torchsummary import summary


class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.rnn = nn.LSTM(
            input_size=1 * 64 * 862,
            hidden_size=64,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,
        )
        self.out = nn.Linear(64, 4)

    def forward(self, x):
        x = self.flatten(x)
        r_out, _ = self.rnn(x, None)
        out = self.out(r_out)
        return out


if __name__ == "__main__":
    cnn = RNN()
    summary(cnn.cuda(), (1, 64, 862))
