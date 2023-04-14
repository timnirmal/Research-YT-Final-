import pandas as pd
import json

# show all columns
pd.set_option('display.max_columns', None)

#
file_name = "filtered_frames.json"
# file_name = "YouTube_post_structure.json"
#
# # load json file
# df = pd.read_json(file_name)


data = json.load(open(file_name))

print(data)

# df = pd.DataFrame(data["properties"])

# print(df)

# read the "Frame:xxx" column Frame as index
df = pd.read_json(file_name, orient='index')

print(df)


# save to csv with index as "xxx"
df.to_csv("data/" + file_name + ".csv", index=True)

# load csv file
df = pd.read_csv("data/" + file_name + ".csv")

print(df)

# from "Unnamed: 0" to "Frame" and remove "Frame:xxx" to "xxx"
df = df.rename(columns={"Unnamed: 0": "frame"})
df['frame'] = df['frame'].str.replace('Frame:', '')
df['frame'] = df['frame'].astype(int)

print(df)

# other column to one column with "Frame" as index
df = df.melt(id_vars=['frame'], var_name='Column', value_name='Value')
# remove null
df = df.dropna()

# remove [ and ]
df['Value'] = df['Value'].str.replace('[', '')
df['Value'] = df['Value'].str.replace(']', '')
# remove '
df['Value'] = df['Value'].str.replace("'", '')

# separate "Value" column to "age", "gen", "emotion"
df[['age', 'gen', 'emotion']] = df['Value'].str.split(',', expand=True)

# sort by frame
df = df.sort_values(by=['frame'])

# remove value,column columns
df = df.drop(columns=['Value', 'Column'])

# age to
df['age'] = df['age'].str.replace('age:', '')

print(df)

# save to csv with index as "xxx"
df.to_csv("data/" + file_name + "-processed.csv", index=False)