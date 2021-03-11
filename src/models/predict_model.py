from starlette.responses import JSONResponse
import pandas as pd
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from category_encoders.binary import BinaryEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from joblib import dump, load

from src.data.sets import save_sets, load_sets
from src.visualization.visualize import classification_reports
from src.models.pytorch import PytorchClassification
from src.models.pytorch import get_device
from src.models.pytorch import train_classification
from src.models.pytorch import test_classification
from src.models.pytorch import PytorchDataset
from src.models.pipes import create_preprocessing_pipe
from src.models.pipes import load_preprocessing_pipe
from src.models.pipes import load_label_encoder
from src.visualization.visualize import plot_nn_model


def predict(model,
            X: pd.DataFrame,
            pipe_name: str = 'pipe.sav',
            label_encoder_name: str = 'label_encoder.sav'):
    pipe = load_preprocessing_pipe(pipe_name=pipe_name)
    X_trans = pipe.transform(X)
    device = get_device()
    X_tensor = torch.Tensor(np.array(X_trans)).to(device)
    le = load_label_encoder()

    preds = model(X_tensor).tolist()

    return JSONResponse(preds)