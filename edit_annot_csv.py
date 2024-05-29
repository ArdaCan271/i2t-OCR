import pandas as pd
from tqdm import tqdm

# Load the CSV file into a DataFrame
df = pd.read_csv('annot.csv')

# Initialize tqdm progress bar
tqdm.pandas()

# Filter out rows containing "a183473b9b4bc8f6" in any column with progress bar
df_filtered = df[~df.progress_apply(lambda row: row.astype(str).str.contains('a183473b9b4bc8f6').any(), axis=1)]

# Save the filtered DataFrame back to the CSV file
df_filtered.to_csv('annot_filtered.csv', index=False)

