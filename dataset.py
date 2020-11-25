import torch.utils.data as data


class AdultDataset(data.Dataset):

    def __init__(self, X, y):

        pass
        ######

        self.X = X
        self.y = y

        ######

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        pass
        ######

        X_out = self.X[index]
        y_out = self.y[index]

        return X_out,y_out

        ######
