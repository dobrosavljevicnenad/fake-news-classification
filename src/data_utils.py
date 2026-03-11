import pandas as pd
import re


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()


def load_welfake_dataset(path="../data/raw/WELFake_Dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")

    df = df[~((df["title"] == "") & (df["text"] == ""))].copy()

    df["content"] = df["title"] + " " + df["text"]
    df["content_clean"] = df["content"].apply(clean_text)

    return df


def print_basic_info(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print(df[["content_clean", "label"]].head())
    print("Label distribution:\n", df["label"].value_counts())