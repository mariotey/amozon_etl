import os
import pandas as pd
import numpy as np
from supabase import create_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPBASE_SECRET_KEY")

# Create client once
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

def clean(v):
    if pd.isna(v):
        return None

    if isinstance(v, (np.integer, np.floating)):
        return v.item()

    return v

def load_to_supabase(df, table_name):
    # Wipe old data first
    supabase_client.rpc(f"truncate_{table_name.lower()}").execute()

    # Insert fresh data
    modified_df = df.copy()

    # Fix invalid JSON values
    modified_df = modified_df.replace([np.inf, -np.inf], None)
    modified_df = modified_df.where(pd.notnull(modified_df), None)

    # Fix datetime values
    for col in modified_df.select_dtypes(include=["datetime64"]).columns:
        modified_df[col] = modified_df[col].astype(str)

    records = [
        {k: clean(v) for k, v in row.items()}
        for row in modified_df.to_dict("records")
    ]

    # Save fresh data
    supabase_client.table(table_name).insert(records).execute()