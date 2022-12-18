from torch import nn
from torchsummary import summary


class FNNNetwork(nn.Module):

    def __init__(self, input_layer_size, layer1_size, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(input_layer_size, layer1_size),
            nn.ReLU(),
            nn.Linear(layer1_size, num_classes)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.flatten(input_data)
        logits = self.dense_layers(x)
        predictions = self.softmax(logits)
        return predictions


if __name__ == "__main__":
    fnn = FNNNetwork(1 * 64 * 431, 256 * 2, 4)
    summary(fnn.cuda(), (1, 64, 431))
