import pandas as pd

from audio.process_data import process_data_df

# load csv
df = pd.read_csv("Process_Audio/data/sinhala-hate-speech-dataset-processed-2.csv")

# show all columns
pd.set_option('display.max_columns', None)
#
# # column_to_process = "Text_cleaned"
# column_to_process = "comment"
#
# # keep only Text_cleaned, label
# df = df[[column_to_process, "label"]]
#
# print(df.head())
#
# # get unique words from Text_cleaned
# unique_words = []
#
# for i in range(df.shape[0]):
#     a = str(df[column_to_process][i]).split(" ")
#     # remove empty string
#     a = list(filter(None, a))
#     unique_words.extend(a)
#
# # add to df
# df = pd.DataFrame(unique_words, columns=["word"])
#
# # remove ´ from words
# df["word"] = df["word"].str.replace("´", "")
#
# print(df.head())
#
# # save csv
# df.to_csv("unique_words_0.csv", index=False)
#
# # remove null item from list
# unique_words = list(filter(None, unique_words))
#
# df_unique_words = process_data_df(df, "word")
#
# # save csv
# df_unique_words.to_csv("unique_words_1.csv", index=False)
#
# # add to df as processed word
# df = pd.DataFrame(df_unique_words, columns=["word"])
#
# # save csv
# df.to_csv("unique_words_2.csv", index=False)
#
#

# load text-1
df = pd.read_csv("unique_words_1.csv")

# remove null
df = df.dropna()
# remove duplicates
df = df.drop_duplicates()

# save csv
df.to_csv("unique_words_4.csv", index=False)

# sort by ascending order in sinhala
df = df.sort_values(by=["word"], ascending=True)

df_0 = pd.read_csv("unique_words_2.csv")

# remove symbols
df_0["word"] = df_0["word"].str.replace("´", "")
df_0["word"] = df_0["word"].str.replace("!", "")
df_0["word"] = df_0["word"].str.replace("?", "")
df_0["word"] = df_0["word"].str.replace(":", "")
df_0["word"] = df_0["word"].str.replace(";", "")
df_0["word"] = df_0["word"].str.replace("(", "")
df_0["word"] = df_0["word"].str.replace(")", "")
df_0["word"] = df_0["word"].str.replace("“", "")
df_0["word"] = df_0["word"].str.replace("”", "")
df_0["word"] = df_0["word"].str.replace("‘", "")
df_0["word"] = df_0["word"].str.replace("’", "")
df_0["word"] = df_0["word"].str.replace("…", "")
df_0["word"] = df_0["word"].str.replace("—", "")
df_0["word"] = df_0["word"].str.replace("–", "")
df_0["word"] = df_0["word"].str.replace("।", "")
df_0["word"] = df_0["word"].str.replace("॥", "")
df_0["word"] = df_0["word"].str.replace("#", "")
df_0["word"] = df_0["word"].str.replace('"', "")
df_0["word"] = df_0["word"].str.replace(",", "")
df_0["word"] = df_0["word"].str.replace(".", "")
df_0["word"] = df_0["word"].str.replace("&", "")
df_0["word"] = df_0["word"].str.replace("'", "")
df_0["word"] = df_0["word"].str.replace("*", "")
df_0["word"] = df_0["word"].str.replace("+", "")
df_0["word"] = df_0["word"].str.replace("-", "")
df_0["word"] = df_0["word"].str.replace("/", "")
df_0["word"] = df_0["word"].str.replace("%", "")
df_0["word"] = df_0["word"].str.replace("`", "")
df_0["word"] = df_0["word"].str.replace("<", "")
df_0["word"] = df_0["word"].str.replace("=", "")
df_0["word"] = df_0["word"].str.replace("@", "")
df_0["word"] = df_0["word"].str.replace("_", "")
df_0["word"] = df_0["word"].str.replace("~", "")
df_0["word"] = df_0["word"].str.replace(" ", "")
# remove numbers with regex
df_0["word"] = df_0["word"].str.replace("\d+", "")
# remove english letters with regex
df_0["word"] = df_0["word"].str.replace("[a-zA-Z]", "")



# count duplicates
df_0 = df_0.groupby(["word"]).size().reset_index(name='counts')

# sort by descending order in sinhala
df_0 = df_0.sort_values(by=["counts"], ascending=False)

# remove first row
df_0 = df_0.iloc[1:]

df_0 = df_0[['counts' , 'word']]

print(df_0.head(40))


# save to csv
df_0.to_csv("unique_words_5.csv", index=False)

# save to xlsx
df_0.to_excel("unique_words_5.xlsx", index=False)







# for i in range(len(unique_words)):
#     unique_words[i] = process_data_str(unique_words[i])
#
# # remove null item from list
# unique_words = list(filter(None, unique_words))
#
# print(len(unique_words))
#
# # save unique words to csv
# df = pd.DataFrame(unique_words, columns=["word"])
# df.to_csv("unique_words.csv", index=False)
#
# # sort by count of each word in descending order
# unique_words = pd.Series(unique_words).value_counts().sort_values(ascending=False)
#
# print(unique_words)

