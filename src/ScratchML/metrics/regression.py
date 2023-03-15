from ScratchML._model_types import PredictionModel
from pandas import DataFrame
from numpy import mean, sum, square, subtract, unique, ndarray

def common_metrics(y_true, y_pred):
    
    return {
        "mse": square(subtract(y_true,y_pred)).mean(),
        "r2": 1 - (square(subtract(y_true,y_pred)).sum() / square(subtract(y_true,mean(y_true))).sum()),
        "mae": abs(subtract(y_true,y_pred)).mean()
    }