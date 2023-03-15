from ScratchML._model_types import PredictionModel
from pandas import DataFrame

def confusion_matrix(y_true, y_pred, labelName, cls_x):
    """
    Generate the confusion matrix

    Args:
        y_true (list): the true labels
        y_pred (list): the predicted labels
        labelName (str): the label name
        cls_x (str): the class x
        
    Returns:
        dict: the confusion matrix
    """
    
    TP = 0 # TRUE POSITIVE
    TN = 0 # TRUE NEGATIVE
    FP = 0 # FALSE POSITIVE
    FN = 0 # FALSE NEGATIVE
    
    df = DataFrame({"true": y_true, "predicted": y_pred})
    
    df["true"] = df["true"] \
        .apply(lambda x: 1 if x == cls_x else 0) # one-hot encoding
    df["predicted"] = df["predicted"] \
        .apply(lambda x: 1 if x == cls_x else 0) # one-hot encoding
    
    PP = df[df["true"] == 1] # Positive Predicted
    TP = len(PP[PP["predicted"] == 1]) # True Positive
    FP = len(PP[PP["predicted"] == 0]) # False Positive
    
    NN = df[df["true"] == 0] # Negative Predicted 
    TN = len(NN[NN["predicted"] == 0]) # True Negative
    FN = len(NN[NN["predicted"] == 1]) # False Negative
    
    
    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN}

def common_metrics(y_true, y_pred, labelName, cls_x):
    """
    Calculate the accuracy

    Args:
        y_true (list): the true labels
        y_pred (list): the predicted labels
        labelName (str): the label name
        cls_x (str): the class x
        
    Returns:
        dict: the common metrics dictionary, containing:
            - accuracy (float): the accuracy
            - recall (float): the recall
            - specificity (float): the specificity
            - precision (float): the precision
            - f1_score (float): the f1-score
            - auc (float): the auc
            
    """
    
    cm = confusion_matrix(y_true, y_pred, labelName, cls_x)
    
    
    return {
        "accuracy": (cm["TP"]+cm["TN"])/(cm["TP"]+cm["TN"]+cm["FP"]+cm["FN"]),
        "recall": cm["TP"]/(cm["TP"]+cm["FN"]),
        "specificity": cm["TN"]/(cm["TN"]+cm["FP"]),
        "precision": cm["TP"]/(cm["TP"]+cm["FP"]),
        "f1_score": 2*cm["TP"]/(2*cm["TP"]+cm["FP"]+cm["FN"]),
        "auc": (cm["TP"]+0.5*cm["FP"])/(cm["TP"]+cm["FN"]+cm["FP"]),
    }