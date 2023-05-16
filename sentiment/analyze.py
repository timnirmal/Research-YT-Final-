import pandas as pd
from matplotlib import pyplot as plt

# load csv
df = pd.read_csv("gossip_lanka_tagged_comments.csv")
df_2 = pd.read_csv("lankadeepa_tagged_comments.csv")

# # show all columns
# pd.set_option('display.max_columns', None)
#
# # remove last column from df
# df = df.iloc[:, :-1]
#
# # rename column
# df = df.rename(columns={'docid': 'comment', 'comment': 'label'})
#
# df_2 = df_2[['comment', 'label']]
#
# # save csv
# df.to_csv("gossip_lanka_tagged_comments.csv", index=False)
#
# # save csv
# df_2.to_csv("lankadeepa_tagged_comments.csv", index=False)

# plot by label
df['label'].value_counts().plot(kind='bar')
plt.show()
df_2['label'].value_counts().plot(kind='bar')
plt.show()

# join df and df_2
df = pd.concat([df, df_2], ignore_index=True)

# in labels 2 -> 1, 3 -> 2, 4 -> 3, 5 -> 4
df['label'] = df['label'].replace([2, 3, 4, 5], [1, 2, 3, 4])

# label 1 keep 3000
df_1 = df[df['label'] == 1].sample(n=3000, random_state=42)
# remove label 1 from df
df = df[df['label'] != 1]
# join df and df_1
df = pd.concat([df, df_1], ignore_index=True)
# shuffle df
df = df.sample(frac=1).reset_index(drop=True)

# plot by label
df['label'].value_counts().plot(kind='bar')
plt.show()


# save csv
df.to_csv("tagged_comments.csv", index=False)