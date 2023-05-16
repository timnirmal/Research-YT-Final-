import pandas as pd
from fuzzywuzzy import fuzz

from audio.process_data import process_data_str



def convert_row(row):
    # df fromat word, start_time, end_time, speaker, text, confidence
    return (row[0], {"start_time": row[1], "end_time": row[2], "speaker": row[3], "text": row[4], "confidence": row[5]})


def process_with_timestamp(df):
    new_df = pd.DataFrame(columns=['word', 'start_time', 'end_time', 'speaker', 'text', 'confidence'])

    # for each unique text in df prcoess_data_str
    for text in df['text'].unique():
        # get words from df where text column equals to text
        word_df = df[df['text'] == text]
        word_df = word_df.apply(convert_row, axis=1)
        unprocessed_data = word_df.tolist()

        print("-"*50)
        processed_data: list[str] = process_data_str(text)
        # list to string
        processed_data = " ".join(processed_data)
        # print(processed_data)
        # print(processed_data)
        # print(processed_data)
        # print(processed_data)
        print("-"*5)
        print(text)

        # Split processed data into words
        words = processed_data.split()

        # Map processed data to unprocessed data using fuzzy string matching
        mapped_data = []
        for word in words:
            # Find closest match for word in unprocessed data
            best_match = max(unprocessed_data, key=lambda x: fuzz.ratio(word, x[0]))
            print(word, " : \t", best_match)
            print("bst0 : ", best_match[0])
            # best match to dataframe
            best_match = pd.DataFrame(best_match[1], index=[0])
            # add word to dataframe
            best_match['word'] = word
            # word to first column
            cols = best_match.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            best_match = best_match[cols]

            print("bst1 : ", best_match)
            # add each word along with metadata to new_df with concat
            # new_df = new_df.append(best_match[1], ignore_index=True)

            # Add metadata to mapped data
            # mapped_data.append(best_match[1])
            # concat to new_df
            new_df = pd.concat([new_df, best_match], ignore_index=True)

        # print(mapped_data)


    print(new_df.tail(10))

    return new_df








