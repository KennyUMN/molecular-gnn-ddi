import random
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, recall_score

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    except ImportError:
        return "cpu"

def calculate_metrics(y_true, y_pred_prob, threshold=0.5):
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    y_pred_binary = (y_pred_prob >= threshold).astype(int)

    # Fail loud: a degenerate evaluation set (one class) has no defined AUROC.
    # Scoring it 0.5 looks like a valid chance-level result and can select a
    # checkpoint from a broken signal (audit finding).
    if len(np.unique(y_true)) < 2:
        raise ValueError(f"degenerate evaluation set: only one class present (n={len(y_true)})")

    auroc = roc_auc_score(y_true, y_pred_prob)
    auprc = average_precision_score(y_true, y_pred_prob)

    acc = accuracy_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall)
    }


