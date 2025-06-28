import numpy as np
import os


def load_dataset(DATA_DIR):
    train = np.load(os.path.join(DATA_DIR, 'picked_aligned_train.npz'), allow_pickle=True)
    val = np.load(os.path.join(DATA_DIR, 'picked_aligned_val.npz'), allow_pickle=True)
    test = np.load(os.path.join(DATA_DIR, 'picked_aligned_test.npz'), allow_pickle=True)

    return (train['X0'].astype(np.float32), train['X1'].astype(np.float32)), train['y'].astype(np.float32), \
        (val['X0'].astype(np.float32), val['X1'].astype(np.float32)), val['y'].astype(np.float32), \
        (test['X0'].astype(np.float32), test['X1'].astype(np.float32)), test['y'].astype(np.float32), \
        test['X0_plot'].astype(np.float32)

def load_calibration_dataset(DATA_DIR):
    cal = np.load(os.path.join(DATA_DIR, 'picked_aligned_calibration.npz'), allow_pickle=True)

    return (cal['X0'].astype(np.float32), cal['X1'].astype(np.float32)), cal['y'].astype(np.float32)

def normalize_bounds(x_train, x_test, x_val=None):
    def get_min_max(idx):
        arrays = [x_train[idx], x_test[idx]]
        if x_val is not None:
            arrays.append(x_val[idx])
        concatenated = np.concatenate(arrays, axis=0)
        return np.min(concatenated, axis=0), np.max(concatenated, axis=0)

    branch_min, branch_max = get_min_max(0)
    trunk_min, trunk_max = get_min_max(1)

    return {
        "branch_min": branch_min,
        "branch_max": branch_max,
        "trunk_min": trunk_min,
        "trunk_max": trunk_max,
    }


def apply_overrides(cfg, overrides):
    """Allow overrides."""
    for kv in overrides or []:
        key, val = kv.split("=")
        # Infer type from config field
        orig_val = getattr(cfg, key)

        # Check if it's a comma-separated list
        if "," in val:
            try:
                # Try to parse as list of floats or ints
                elem_type = type(orig_val[0]) if isinstance(orig_val, list) and orig_val else float
                val = [elem_type(v) for v in val.split(",")]
            except Exception:
                pass  # Fallback
        elif isinstance(orig_val, bool):
            val = val.lower() in ("true", "1")
        elif isinstance(orig_val, float):
            val = float(val)
        elif isinstance(orig_val, int):
            val = int(val)
        elif val.lower() == "null":
            val = None
        setattr(cfg, key, val)


def transform_input(x, min_val, max_val):
    d = x.shape[1]
    x = 2 * (x - min_val) / (max_val - min_val) - 1
    x = x / np.sqrt(d)
    x_d1 = np.sqrt(1 - np.sum(x**2, axis=1, keepdims=True))
    return np.concatenate((x, x_d1), axis=1)