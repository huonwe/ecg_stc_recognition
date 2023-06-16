import numpy as np
import torch


class Navigator:
    def __init__(self, threhod=0.0, method="dist"):
        self.threshold = threhod
        self.library = []
        self.method = method

    def register(self, label, feature):
        if self.search(feature) != -1:
            self.library.append({'label': label, 'feature': feature})
        else:
            return -1

    def identify(self, feature):
        return self.search(feature)

    def search(self, feature):
        if self.method == "dist":
            for unit in self.library:
                distance = self.distance(feature, unit['feature'])
                if distance < self.threshold:
                    return unit['label']
        elif self.method == "cos_sim":
            pass
        return -1

    def load(self, library):
        self.library = library

    def clear(self):
        self.library = []

    @staticmethod
    def distance(feature1, feature2):
        diff = torch.subtract(feature1, feature2)
        dist = torch.sum(torch.square(diff))
        return dist.item()

    @staticmethod
    def cos_sim(a, b):
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        cos = np.dot(a, b) / (a_norm * b_norm)
        return cos

    def same(self, feature1, feature2):
        if self.method == "dist":
            dist = self.distance(feature1, feature2)
            if dist < self.threshold:
                return True, dist
            else:
                return False, dist

        elif self.method == "cos_sim":
            pass
