import time

import pandas as pd

df_lexicon = pd.read_csv("sentiment/data/lexicon.csv")

# convert to dict
lexicon = df_lexicon.set_index('word')['sentiment'].to_dict()

def calculate_sentiment(text, sentiment_dict=lexicon):
    words = text.lower().split()
    sentiment_sum = 0
    sentiment_count = 0

    for word in words:
        if word in sentiment_dict:
            sentiment_sum += sentiment_dict[word]
            sentiment_count += 1

    return sentiment_sum / sentiment_count if sentiment_count > 0 else 0


def calculate_sentiment_df(df):
    # for each row in df
    for index, row in df.iterrows():
        # get text
        text = row["word"]

        # calculate sentiment
        sentiment = calculate_sentiment(text)

        # add sentiment to df
        df.loc[index, "sentiment_word"] = sentiment

        # save csv
        df.to_csv("recognized_processed_sentiment_word.csv", index=False)

    return df



