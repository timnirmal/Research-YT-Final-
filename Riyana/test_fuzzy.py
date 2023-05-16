import pandas as pd
from fuzzywuzzy import fuzz

# # Unprocessed data with metadata
# unprocessed_data = [("apple", {"color": "red", "type": "fruit"}),
#                     ("banana", {"color": "yellow", "type": "fruit"}),
#                     ("carrot", {"color": "orange", "type": "vegetable"})]

df = pd.read_csv("Research/Codes/Ex05.csv")

def convert_row(row):
    # df fromat word, start_time, end_time, speaker, text, confidence
    return (row[0], {"start_time": row[1], "end_time": row[2], "speaker": row[3], "text": row[4], "confidence": row[5]})

# convert each row in df to tuple
df = df.apply(convert_row, axis=1)

# convert df to list
unprocessed_data = df.to_list()

print(unprocessed_data)



unprocessed_data = [
    ("This", {"start": 0, "end": 1}),
    ("is", {"start": 2, "end": 3}),
    ("an", {"start": 4, "end": 5}),
    ("example", {"start": 6, "end": 13}),
    ("of", {"start": 14, "end": 15}),
    ("processed", {"start": 16, "end": 25}),
    ("data", {"start": 26, "end": 30}),
    ("with", {"start": 31, "end": 35}),
    ("missing", {"start": 36, "end": 43}),
    ("characters", {"start": 44, "end": 54})
]



# Processed data with missing characters
# processed_data = "Ths is an exmple of prcessed missng charcters"
processed_data = "මගේ දැනුමට කරන්ඩ ඕනේ ඒක මට අදාළ කාරණාවක් නෑ මේකට varadar ඉන්නවා හැබැයි ක්‍රිකට් ක්‍රීඩාව ජයග්‍රහණය කරලා වන ඔබතුමාලා පිළිගන්න ඕන තරම් ඉන්න කට්ටිය mal mala dala හැබැයි මේ වෙද්දි කවුරුවත් නැහැ රහස් එකම යථාර්තය නම් මේ බලධාරීන් බලධාරීන් තමයි මාධ්‍ය නියමයි ජයග්‍රහණය කර මරදාන කට්ටිය පරාජය වගේ තමයි මට විශේෂයම කියන්ට තියුණු වැල වීඩියෝ වීඩියෝ"

# Split processed data into words
words = processed_data.split()

# Map processed data to unprocessed data using fuzzy string matching
mapped_data = []
for word in words:
    # Find closest match for word in unprocessed data
    best_match = max(unprocessed_data, key=lambda x: fuzz.ratio(word, x[0]))
    print(word, " : \t", best_match)
    # Add metadata to mapped data
    mapped_data.append(best_match[1])

print(mapped_data)
