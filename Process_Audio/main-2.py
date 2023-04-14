import pandas as pd

from lib.process_data import process_data_df

# load csv file
df = pd.read_csv("data/sinhala-hate-speech-dataset.csv")

print(df.head())

result = process_data_df(df)



print("Done!")
print("Done!")
print("Done!")
# print(result)

# # load csv file
# result = pd.read_csv("data/sinhala-hate-speech-dataset-processed.csv")

# remove null values
result = result.dropna()

# remove if only whitespace
result = result[~result['Text_cleaned'].str.isspace()]

# save to csv
result.to_csv("data/sinhala-hate-speech-dataset-processed-2.csv", index=False)

# keep only the columns we need
result = result[['Text_cleaned', 'label']]
result = result.rename(columns={"Text_cleaned": "text", "label": "label"})
# switch column order
result = result[['text', 'label']]
# save to csv
result.to_csv("data/sinhala-hate-speech-dataset-processed-3.csv", index=False)

# remove null values
result = result.dropna()

print(result.head())

def analyze_dataset():
    # get number of rows
    print("Number of rows: " + str(len(result.index)))
    # number of rows with label 0
    print("Number of rows with label 0: " + str(len(result[result['label'] == 0].index)))
    # number of rows with label 1
    print("Number of rows with label 1: " + str(len(result[result['label'] == 1].index)))


analyze_dataset()