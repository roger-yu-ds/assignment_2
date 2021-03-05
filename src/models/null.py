import pandas as pd
import numpy as np
from statistics import mode
from sklearn.preprocessing import LabelEncoder


class NullModel():
    def __init__(self, target_type: str = 'regression'):
        self.target_type = target_type
        self.y = None
        self.pred_value = None
        self.preds = None
        self.le = None

    def fit(self, y):
        self.y = y
        if self.target_type == 'regression':
            self.pred_value = y.mean()
        else:
            if y.nunique() > 2:
                self.le = LabelEncoder()
                self.le.fit(y)
                y_enc = self.le.transform(y)
                self.pred_value = mode(y_enc)

    def get_length(self, y):
        return len(self.y)

    def predict(self, y):
        self.preds = [self.pred_value] * self.get_length(y)

        if self.le is not None:
            self.preds =


        return self.preds

    def fit_predict(self, y):
        self.fit(y)
        return self.predict(y)