# filter_precursor_range_script.py
import pandas as pd

def filter_precursor_range(
    df: pd.DataFrame,
    min_mz: float = None,
    max_mz: float = None
) -> pd.DataFrame:
    """
    Filter a hits dataframe to keep only rows within a precursor m/z range.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with a `precmz` column.
    min_mz : float, optional
        Minimum precursor m/z value to keep (inclusive).
    max_mz : float, optional
        Maximum precursor m/z value to keep (inclusive).
    
    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    """
    if "precmz" not in df.columns:
        raise KeyError("Dataframe must contain a 'precmz' column.")
    
    filtered = df.copy()
    if min_mz is not None:
        filtered = filtered[filtered["precmz"] >= min_mz]
    if max_mz is not None:
        filtered = filtered[filtered["precmz"] <= max_mz]
    
    return filtered
