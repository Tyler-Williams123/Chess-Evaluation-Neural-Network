import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, inDim, outDim, hiddenSize):
        super().__init__()

        self.layer1 = nn.Linear(in_features=inDim, out_features=hiddenSize)
        self.layer2 = nn.GELU()
        self.layer3 = nn.Linear(in_features=hiddenSize, out_features=hiddenSize)
        self.layer4 = nn.GELU()
        self.layer5 = nn.Linear(in_features=hiddenSize, out_features=hiddenSize)
        self.layer6 = nn.GELU()
        self.layer7 = nn.Linear(in_features=hiddenSize, out_features=outDim)

    def forward(self, data):
        prediction = self.layer2(self.layer1(data))
        prediction = self.layer4(self.layer3(prediction))
        prediction = self.layer6(self.layer5(prediction))
        prediction = self.layer7(prediction)

        return prediction