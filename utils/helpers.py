import pandas as pd


def downcast_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(
        lambda col: pd.to_numeric(col, downcast="float")
        if col.dtype.kind == "f"
        else pd.to_numeric(col, downcast="integer")
        if col.dtype.kind in "iu"
        else col
    )
