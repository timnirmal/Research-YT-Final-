import pandas as pd
from audio.process_data import process_data, process_data_df

# load csv
df = pd.read_csv("sentiment/tagged_comments.csv")

# show all columns
pd.set_option('display.max_columns', None)

# process
df = process_data_df(df)

# save csv
df.to_csv("sentiment/tagged_comments_processed.csv", index=False)