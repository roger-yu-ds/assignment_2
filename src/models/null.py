import pandas as pd
import numpy as np
from statistics import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class NullModel():
    def __init__(self, target_type: str = 'classification'):
        self.target_type = target_type
        self.y = None
        self.pred_value = None
        self.preds = None
        self.ohe = OneHotEncoder()
        self.le = LabelEncoder()

    def fit(self, y):
        self.y = y
        if self.target_type == 'regression':
            self.pred_value = y.mean()
        else:
            if y.nunique() > 2:
                self.ohe.fit(pd.DataFrame({'y': y}))
                self.pred_value = mode(y)

    def get_length(self, y):
        return len(self.y)

    def predict(self, y):
        preds_df = pd.DataFrame({'preds': [self.pred_value] * self.get_length(y)})
        self.preds = self.ohe.transform(preds_df)

        return self.preds

    def fit_predict(self, y):
        self.fit(y)
        return self.predict(y)
