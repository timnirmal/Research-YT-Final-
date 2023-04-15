import pandas as pd

from Binary_model import predict_from_best, train

# load csv file
df = pd.read_csv("../Riyana/Research/Text-Classification-Module-master/Dataset/cleaned data-2.csv")


# remove null values
df = df.dropna()
# remove duplicates
df = df.drop_duplicates(subset=['Filtered_sentence'], keep='first')
# remove if only whitespace
df = df[~df['Filtered_sentence'].str.isspace()]
# remove leading and trailing whitespace
df['Filtered_sentence'] = df['Filtered_sentence'].str.strip()

# create copy of df
df_copy = df.copy()

df_copy_2 = df.copy()

# keep only Text_cleaned and Label columns
df_copy = df_copy[['Filtered_sentence']]

df_copy_2 = df_copy_2[['Text']]

# train model
best_model = train()

# for each row in df_copy
for index, row in df_copy.iterrows():
    # predict for each text
    prediction = predict_from_best(row['Filtered_sentence'], best_model)
    # add prediction to df_copy
    df.at[index, 'Prediction'] = prediction

# for each row in df_copy_2
for index, row in df_copy_2.iterrows():
    # predict for each text
    prediction = predict_from_best(row['Text'], best_model)
    # add prediction to df_copy
    df.at[index, 'Prediction_2'] = prediction

print(df.head())
# save to csv
df.to_csv("data/sinhala-hate-speech-dataset-processed-4.csv", index=False)
