from datasets import load_dataset
from config import DATA_NAME, DATA_CATEGORY
import pandas as pd
import ast
from tqdm import tqdm
tqdm.pandas

def extract_review_data():
    review_dataset = load_dataset(
        DATA_NAME,
        f"raw_review_{DATA_CATEGORY}"
    )

    all_reviews = [review for review in tqdm(review_dataset["full"])]
    review_df = pd.DataFrame(all_reviews)

    modified_review_df = review_df.copy()

    modified_review_df["rating"] = pd.to_numeric(modified_review_df["rating"], errors="coerce")
    modified_review_df["title"] = modified_review_df["title"].str.strip().str.lower()
    modified_review_df["text"] = modified_review_df["text"].str.strip().str.lower()
    modified_review_df["images"] = modified_review_df["images"].astype(str)
    modified_review_df["asin"] = modified_review_df["asin"].str.lower()
    modified_review_df["parent_asin"] = modified_review_df["parent_asin"].str.lower()
    modified_review_df["user_id"] = modified_review_df["user_id"].str.lower()
    modified_review_df["date"] = pd.to_datetime(modified_review_df["timestamp"], unit="ms").dt.date

    # Reorder columns
    modified_review_df = modified_review_df[[
        "asin",
        "parent_asin",
        "user_id",
        # "timestamp",
        "date",
        "title",
        "text",
        "images",
        "verified_purchase",
        "helpful_vote",
        "rating"
    ]]

    # Rename columns
    modified_review_df = modified_review_df.rename(columns={
        "date": "review_date",
        "title": "review_title",
        "text": "review_text",
        "images": "review_images",
        "rating": "review_rating"
    })

    return modified_review_df

def extract_meta_data():
    meta_dataset = load_dataset(
        DATA_NAME,
        f"raw_meta_{DATA_CATEGORY}",
        split="full"
    )

    meta_df = pd.DataFrame(meta_dataset)

    category_replace_dict = {
        "accounting": "accounting & finance",
        "antivirus": "antivirus & security",
        "education": "education & reference",
        "free one-day shipping i software": "free one-day shipping for software",
        "free one-day shipping on select software with your citi card": "free one-day shipping for software",
        "medicine": "medicine & health sciences",
        "photography": "photography & graphic design",
        "programming": "programming & web development",
        "spreadsheet": "spreadsheet & database",
        "training": "training & tutorials"
    }

    def parse_categories(cat_str):
        """Parse string representation of list into actual list"""
        if pd.isna(cat_str) or cat_str.strip() == "[]":
            return []
        try:
            parsed_list = ast.literal_eval(cat_str)

            modified_list = []

            for elem in parsed_list:
                modified_elem = (
                    elem
                    .lower()
                    .replace("\'s", "")
                )

                if modified_elem in category_replace_dict.keys():
                    modified_list.append(category_replace_dict[modified_elem])
                else:
                    modified_list.append(modified_elem)

            return sorted(modified_list)

        except Exception as e:
            print(f"{e}: {cat_str}")
            return [cat_str]

    def parse_videos(video_dict):
        for key, val in video_dict.items():
            if val == [""]:
                video_dict[key] = []

        return video_dict

    modified_meta_df = meta_df.copy()

    modified_meta_df["parent_asin"] = modified_meta_df["parent_asin"].str.lower()
    modified_meta_df["title"] = modified_meta_df["title"].str.strip().str.lower()
    modified_meta_df["main_category"] = modified_meta_df["main_category"].str.strip().str.lower()
    modified_meta_df["categories"] = modified_meta_df["categories"].astype(str).apply(parse_categories)
    modified_meta_df["videos"] = modified_meta_df["videos"].apply(parse_videos)

    # modified_meta_df["average_rating"] = pd.to_numeric(modified_meta_df["average_rating"], errors="coerce")
    modified_meta_df["rating_number"] = pd.to_numeric(modified_meta_df["rating_number"], errors="coerce")

    modified_meta_df["store"] = modified_meta_df["store"].astype(str).str.strip().str.lower()
    modified_meta_df["price"] = pd.to_numeric(modified_meta_df["price"], errors="coerce")

    # Reorder columns
    modified_meta_df = modified_meta_df[[
        "parent_asin",
        "title",
        "main_category",
        "categories",
        "description",
        "features",
        "details",
        "images",
        "videos",
        # "bought_together", # All rows contain None
        # "average_rating", # Removing since more concern about rating on platform than product page
        "rating_number",
        "store",
        # "subtitle", # All rows contain None
        # "author", # All rows contain None
        "price"
    ]]

    # Rename columns
    modified_meta_df = modified_meta_df.rename(columns={
        "title": "item_title",
        "images": "item_images",
        "videos": "item_videos",
        "rating_number": "item_rating"
    })

    return modified_meta_df