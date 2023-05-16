import pandas as pd

from audio.process_data import process_data_str

# load csv
df = pd.read_csv("Process_Audio/data/sinhala-hate-speech-dataset-processed-2.csv")

# show all columns
pd.set_option('display.max_columns', None)

# column_to_process = "Text_cleaned"
column_to_process = "comment"

# keep only Text_cleaned, label
df = df[[column_to_process, "label"]]

print(df.head())

# get unique words from Text_cleaned
unique_words = []

for i in range(df.shape[0]):
    a = str(df[column_to_process][i]).split(" ")
    # remove empty string
    a = list(filter(None, a))
    unique_words.extend(a)

# print(unique_words)
print(len(unique_words))

for i in range(len(unique_words)):
    unique_words[i] = process_data_str(unique_words[i])

# remove null item from list
unique_words = list(filter(None, unique_words))

print(len(unique_words))

# save unique words to csv
df = pd.DataFrame(unique_words, columns=["word"])
df.to_csv("unique_words.csv", index=False)

# sort by count of each word in descending order
unique_words = pd.Series(unique_words).value_counts().sort_values(ascending=False)

print(unique_words)
