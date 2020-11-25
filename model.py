import torch.nn as nn


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_size):

        super(MultiLayerPerceptron, self).__init__()

        ######

        self.fc1 = nn.Linear(input_size,64)
        self.fc2 = nn.Linear(64,1)


        self.output = nn.Linear(1,1)

        ######

    def forward(self, features):

        pass
        features = self.fc1(features)
        features = nn.functional.relu(self.fc2(features))

        features = nn.functional.sigmoid(self.output(features))

        return features
        

        ######
