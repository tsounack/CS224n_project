import pandas as pd

# Load the CSV file
df = pd.read_csv('Data/Lakers_QA_Dataset - Sheet1.csv')

filtered_df = df[~df.apply(lambda row: row['answer'].replace('\n', '').replace("  ", " ") in row['Paragraph'].replace('\n', '').replace("  ", " "), axis=1)]

# Now 'df' is a DataFrame holding the content of the CSV file
filtered_df.to_csv('output.csv', index=False)

for index, row in filtered_df.iterrows():
    print(row['Paragraph'].replace('\n',' '))
    print('\n')


