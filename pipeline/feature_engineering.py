import os
import pickle

import pandas as pd
import pathlib
import string

import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from pipeline.pipeline_utils import get_logger, load_config_params

logger = get_logger("feature_engineering")

nltk.download('stopwords')
nltk.download('punkt')
ss = SnowballStemmer('english')
tfidf = TfidfVectorizer(max_features=300)
_stop_words = set(stopwords.words('english'))



def load_data(path: str) -> pd.DataFrame:
    """
    Load data from the path
    :param path: path of file
    :return: Dataframe
    """
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        raise
    except Exception as e:
        logger.error(e)
        raise


def transform_text(text) -> str:
    """
    Transform the text by transforming into lowercase,
    remove the stop words, special characters, emojis and
    do the word steaming
    :param text: Input text
    :return: transformed text
    """
    text = text.lower()

    text = nltk.word_tokenize(text)

    # Removing Special Characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # Removing stop words and punctuations
    text = y[:]
    y.clear()

    for i in text:
        if i not in _stop_words and i not in string.punctuation:
            y.append(i)


    # Stemming using Snowball Stemmer
    text = y[:]
    y.clear()

    for i in text:
        y.append(ss.stem(i))

    return " ".join(y)

def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df['transformed_text'] = df['text'].apply(transform_text)
    return df

def vectorize_dataframe(df: pd.DataFrame):
    tfidf_matrix = tfidf.fit_transform(df['transformed_text'])

    # Create a DataFrame from the TF-IDF array with readable feature column names
    feature_names = tfidf.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{name}" for name in feature_names])

    # Concatenate the original dataframe with the new TF-IDF feature columns
    df_combined = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    return df_combined

def process_all_data(folder_path: str, output_folder: str):
    try:
        os.makedirs(output_folder, exist_ok=True)

        folder = pathlib.Path(folder_path)
        files = [file for file in folder.iterdir() if file.is_file()]
        logger.debug(f"Processing {len(files)} files")

        # Step 1: transform all files first
        dataframes = {}
        for file in files:
            df = load_data(str(file.resolve()))
            df = transform_dataframe(df)
            dataframes[file.stem] = df

        # Step 2: fit tfidf on all data combined so every file shares same vocabulary
        all_text = pd.concat([df['transformed_text'] for df in dataframes.values()])
        tfidf.fit(all_text)
        logger.debug(f"TF-IDF fitted on {len(all_text)} total samples")

        os.makedirs("models", exist_ok=True)
        with open("models/tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(tfidf, f)
        logger.debug("TF-IDF vectorizer saved to models/tfidf_vectorizer.pkl")

        # Step 3: transform each file with shared vocabulary and save
        for name, df in dataframes.items():
            tfidf_matrix = tfidf.transform(df['transformed_text'])
            feature_names = tfidf.get_feature_names_out()
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{n}" for n in feature_names])
            df_combined = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
            df_combined.to_csv(os.path.join(output_folder, name + '.csv'), index=False)
            logger.debug(f"Processed {name}")

        logger.debug(f"Processed all {len(files)} files")

    except Exception as e:
        logger.error(f"Error :{e}")

def main():
    params = load_config_params("params.yaml", logger)
    data_folder = params["utils"]["data_folder"]
    processed_data_path = f"{data_folder}/processed"
    encoded_data_path = f"{data_folder}/encoded"

    process_all_data(processed_data_path, encoded_data_path)

    logger.debug(f"Processed all files")

if __name__ == "__main__":
    main()