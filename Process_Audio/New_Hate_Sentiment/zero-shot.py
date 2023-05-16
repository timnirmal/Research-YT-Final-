# Install transformers library
# !pip install transformers==3.1.0

# Import the Transformers pipeline library
from transformers import pipeline

# Preprocessing and visualization libraries
import plotly.express as px
import pandas as pd
import numpy as np
import textwrap

wrapper = textwrap.TextWrapper(width=80)

# Load the dataset
data_url = "https://raw.githubusercontent.com/keitazoumana/Zero-Shot-Text-Classification/main/bbc-text.csv"
news_data = pd.read_csv(data_url)
news_data.head()

zsmlc_classifier = pipeline("zero-shot-classification", model='joeddav/xlm-roberta-large-xnli')

# Select the description of the first row.
sequences = news_data.iloc[0]["text"]

# Get all the candidate labels
candidate_labels = list(news_data.category.unique())

# Run the result
result = zsmlc_classifier(sequences, candidate_labels, multi_class=True)

# show the result
result

# Delete the sequence key
del result["sequence"]
result_df = pd.DataFrame(result)
result_df

# Plot the probability distributions
fig = px.bar(result_df, x='labels', y='scores')
fig.show()


def make_prediction(clf_result):
    # Get the index of the maximum probability score
    max_index = np.argmax(clf_result["scores"])
    predicted_label = clf_result["labels"][max_index]

    return predicted_label


# print(make_prediction(result))


def select_subset_data(data, label_column, each_target_size=2):
    all_targets = list(data[label_column].unique())
    list_dataframes = []

    for label in all_targets:
        subset = data[data[label_column] == str(label)]
        subset = subset.sample(each_target_size)

        list_dataframes.append(subset)

    return pd.concat(list_dataframes)


def run_batch_prediction(original_data, label_column, desc_col, my_classifier=zsmlc_classifier):
    # Make a copy of the data
    data_copy = original_data.copy()

    # The list that will contain the models predictions
    final_list_labels = []

    for index in range(len(original_data)):
        # Run classification
        sequences = original_data.iloc[index][desc_col]
        candidate_labels = list(original_data[label_column].unique())
        result = my_classifier(sequences, candidate_labels, multi_class=True)

        # Make prediction
        final_list_labels.append(make_prediction(result))

    # Create the new column for the predictions
    data_copy["clf_predictions"] = final_list_labels

    return data_copy


# Get the subset of dataframe
subset_news_data = select_subset_data(news_data, "category")

# Run the predictions on the new dataset
pred_res_data = run_batch_prediction(subset_news_data, "category", "text")
pred_res_data


def show_labels_prediction(data, row_of_interest):
    # Select the description of the first row.
    sequences = data.iloc[221]["text"]

    # Get all the candidate labels
    candidate_labels = list(data.category.unique())

    # Run the result
    result = zsmlc_classifier(sequences, candidate_labels, multi_class=True)

    # Make the result
    result['sequence'] = wrapper.fill(result['sequence'])

    # Show the corresponding text
    print(result["sequence"])

    # Delete the sequence key
    del result["sequence"]

    result_df = pd.DataFrame(result)
    result_df

    # Plot the probability distributions
    fig = px.bar(result_df, x='labels', y='scores')
    fig.show()


show_labels_prediction(news_data, 221)
