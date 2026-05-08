import logging
from etl import extract, transform, load
from config import (
    USER_TABLE_NAME,
    ITEM_TABLE_NAME,
    REVIEW_TABLE_NAME
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    # Extract
    raw_review_df = extract.extract_review_data()
    logger.info(f"\nReview data extracted: {raw_review_df.shape} \n")

    raw_meta_df = extract.extract_meta_data()
    logger.info(f"\nMeta data extracted: {raw_meta_df.shape} \n")

    # Transform
    review_df = transform.get_review_df(raw_review_df)
    logger.info(f"\nReview dataframe created: {review_df.shape} \n")

    user_df = transform.get_user_df(review_df, raw_meta_df)
    logger.info(f"\nUser dataframe created: {user_df.shape} \n")

    item_df = transform.get_item_df(review_df, raw_meta_df)
    logger.info(f"\nItem dataframe created: {item_df.shape} \n")

    # Load
    # load.load_to_supabase(user_df, USER_TABLE_NAME)
    # load.load_to_supabase(item_df, ITEM_TABLE_NAME)
    # load.load_to_supabase(review_df, REVIEW_TABLE_NAME)

if __name__ == "__main__":
    main()