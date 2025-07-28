from __future__ import annotations

def load_ucr_uea(name: str, split: str = None, return_type: str = "numpy3D",
                 extract_path: str | None = None, y_dtype: str = "str"):
    from sktime.datasets import load_UCR_UEA_dataset
    X, y = load_UCR_UEA_dataset(
        name=name,
        split=split,
        return_X_y=True,
        return_type=return_type,
        extract_path=extract_path,
        y_dtype=y_dtype,
    )
    return X, y


def load_tsfile_to_numpy3d(full_ts_path: str,
                           replace_missing_vals_with: str | None = None,
                           encoding: str | None = None):
    try:
        from sktime.datasets import load_from_tsfile
        X, y = load_from_tsfile(
            full_ts_path,
            return_separate_X_and_y=True,
            replace_missing_vals_with=replace_missing_vals_with,
            encoding=encoding,
            return_type="numpy3D", 
        )
        return X, y
    except ImportError:
        from sktime.datasets import load_from_tsfile_to_dataframe
        X_df, y = load_from_tsfile_to_dataframe(
            full_ts_path,
            return_separate_X_and_y=True,
            replace_missing_vals_with=replace_missing_vals_with,
            encoding=encoding,
        )
        import numpy as np
        import pandas as pd

        lengths = {c: X_df.iloc[0, c].shape[0] for c in range(X_df.shape[1])}
        L = list(lengths.values())[0]
        D = X_df.shape[1]
        N = X_df.shape[0]
        X = np.empty((N, D, L), dtype="float32")
        for i in range(N):
            for d in range(D):
                s = X_df.iat[i, d]
                arr = np.asarray(s, dtype="float32")
                if arr.shape[0] != L:
                    out = np.zeros((L,), dtype="float32")
                    m = min(L, arr.shape[0])
                    out[:m] = arr[:m]
                    arr = out
                X[i, d, :] = arr
        return X, y
