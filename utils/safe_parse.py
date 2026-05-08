import numpy as np
import pandas as pd
import json

def safe_len(x):
    """Safely get length of arrays"""
    if isinstance(x, (list, np.ndarray)):
        # Drop duplicated elements in the list
        return len(set(x))
    return 0

def safe_join(x):
    if isinstance(x, (list, np.ndarray)):
        if len(x) == 0:
            return ''
        return ','.join([str(i) for i in x if i is not None])
    return '' if pd.isna(x) else str(x)

def safe_json_numpy(x):
    if isinstance(x, dict):
        return json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in x.items()})
    elif isinstance(x, np.ndarray):
        return json.dumps(x.tolist())
    elif isinstance(x, (list)):
        return json.dumps(x)
    return '' if pd.isna(x) else str(x)