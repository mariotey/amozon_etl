import os
import time
import pandas as pd
import numpy as np
import logging
from supabase import create_client
from dotenv import load_dotenv
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

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

def chunk_list(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield i, data[i:i + batch_size]

def load_to_supabase(
    df,
    table_name,
    batch_size=8000,
    sleep_seconds=0.75
):
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
        for row in tqdm(modified_df.to_dict("records"), desc="Cleaning rows")
    ]

    # Save fresh data
    total = len(records)

    for start, batch in tqdm(
        chunk_list(records, batch_size),
        total=(len(records) + batch_size - 1) // batch_size,
        desc=f"Loading {table_name}"
    ):
        end = start + len(batch)

        try:
            supabase_client.table(table_name).insert(batch).execute()
        except Exception as e:
            logger.error(f"Batch {start}-{end} failed: {e}")
            raise

        time.sleep(sleep_seconds)

    logger.info("✅ Loaded into Supabase successfully")