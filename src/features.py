import numpy as np
import pandas as pd


def clean_raw_columns(df):
    """Fix the raw Kaggle export quirks: fake 'NaN' strings, unit prefixes."""
    df = df.copy()

    # These are numbers stored as text, with 'NaN ' as a fake missing-value string
    for col in ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"]:
        df[col] = df[col].str.strip()
        df[col] = df[col].replace("NaN", np.nan)
        df[col] = pd.to_numeric(df[col])

    # Target column: "(min) 24" -> 24
    df["Time_taken(min)"] = (
        df["Time_taken(min)"].str.extract(r"(\d+)", expand=False).astype(int)
    )

    # Weather: "conditions Sunny" -> "Sunny", "conditions NaN" -> real missing
    df["Weatherconditions"] = df["Weatherconditions"].str.strip().str.replace(
        "conditions ", "", regex=False
    )
    df["Weatherconditions"] = df["Weatherconditions"].replace("NaN", np.nan)

    return df