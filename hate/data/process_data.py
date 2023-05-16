import pandas as pd

# read xlsx
df = pd.read_excel("unique_words_5.xlsx")
# remove first column
df = df.drop(df.columns[0], axis=1)
# save as csv
df.to_csv("unique_words_5.csv", index=False)
# read csv
df = pd.read_csv("unique_words_5.csv")

# show all columns
pd.set_option('display.max_columns', None)

print(df.head())

# # replace 1 with -1
# df = df.replace(1, -1)
#
# # fill NaN with 0
# df = df.fillna(1)

print(df.head())

# save as csv
df.to_csv("unique_words_5.csv", index=False)