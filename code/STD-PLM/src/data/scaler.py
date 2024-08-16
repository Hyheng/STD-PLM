import torch

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        dim = data.shape[-1]
        data = torch.concat([(data[...,i:i+1] - self.mean[i]) / self.std[i] for i in range(dim)],dim=-1)

        return data

    def inverse_transform(self, data):
        dim = data.shape[-1]
        data = torch.concat([(data[...,i:i+1] * self.std[i]) + self.mean[i] for i in range(dim)],dim=-1)
        return data

class MinMaxScaler():
    def __init__(self, min,max):

        self._min = min
        self._max = max

    def transform(self, data):
        data = 1. * (data - self._min)/(self._max - self._min)
        data = data * 2. - 1.
        return data

    def inverse_transform(self, data):
        dim = data.shape[-1]
        data = (data + 1.) / 2.
        data = 1. * data * (self._max[...,:dim] - self._min[...,:dim]) + self._min[...,:dim]
        return data