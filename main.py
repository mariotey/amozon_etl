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
    logger.info(f"\n{review_df.dtypes} \n")

    # user_df = transform.get_user_df(review_df, raw_meta_df)
    # logger.info(f"\nUser dataframe created: {user_df.shape} \n")
    # logger.info(f"\n{user_df.dtypes} \n")

    item_df = transform.get_item_df(review_df, raw_meta_df)
    logger.info(f"\nItem dataframe created: {item_df.shape} \n")
    logger.info(f"\n{item_df.dtypes} \n")
    # Load
    # load.load_to_supabase(
    #     user_df,
    #     USER_TABLE_NAME,
    #     batch_size=8000
    # )
    load.load_to_supabase(
        item_df[[
            "parent_asin", "item_title", "main_category", "categories", "description", "features", "details", "price", "num_item_img", "num_item_videos", "is_free"
        ]],
        ITEM_TABLE_NAME,
        batch_size=1000
    )
    load.load_to_supabase(
        review_df[[
            "review_id", "parent_asin", "user_id", "helpful_vote", "review_rating", "review_word_count", "num_review_img", "recency_weight"
        ]],
        REVIEW_TABLE_NAME,
        batch_size=5000
    )

if __name__ == "__main__":
    main()