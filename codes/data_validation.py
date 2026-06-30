import pandas as pd
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def validate_data_integrity(df: pd.DataFrame, drop_nulls: bool = True, null_threshold: float = 0.05) -> Tuple[bool, pd.DataFrame]:
    """
    Validates a DataFrame for nulls and infinite values.
    Returns (is_valid, cleaned_df).
    """
    if df.empty:
        logger.warning("DataFrame is empty.")
        return False, df

    # Check for infinite values
    inf_cols = df.columns.to_series()[np.isinf(df.select_dtypes(include=[np.number])).any()]
    if not inf_cols.empty:
        logger.error(f"Infinite values found in columns: {inf_cols.tolist()}")
        # Replace inf with NaN for uniform handling
        df = df.replace([np.inf, -np.inf], np.nan)

    # Check for nulls
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    
    if null_cols.empty:
        return True, df
        
    total_samples = len(df)
    logger.warning(f"Null values detected in {len(null_cols)} columns.")
    
    for col, count in null_cols.items():
        null_rate = count / total_samples
        logger.warning(f"  {col}: {count} nulls ({null_rate:.1%})")
        
        if null_rate > null_threshold:
            logger.error(f"Column {col} exceeds null threshold of {null_threshold:.1%}")
            
    if drop_nulls:
        initial_len = len(df)
        df_clean = df.dropna().reset_index(drop=True)
        dropped = initial_len - len(df_clean)
        logger.info(f"Dropped {dropped} rows with null/inf values.")
        return True, df_clean
        
    return False, df
