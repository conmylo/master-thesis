import pandas as pd

# Load the dataset
df = pd.read_csv('data/subSmall.csv', on_bad_lines='skip')

# Display the first few rows and the column names
print(df.head())
print(df.columns)
