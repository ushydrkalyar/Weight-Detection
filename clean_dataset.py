import pandas as pd
import re

# 1. Load the Kaggle dataset
df = pd.read_csv('dataset.csv')

# 2. Define a function to extract and convert the data
def parse_and_convert(text):
    # This regex looks for: (number)' (number)" (number) lbs
    match = re.search(r"(\d+)'\s*(\d+)\"\s*(\d+)\s*lbs\.?", str(text), re.IGNORECASE)
    
    if match:
        feet = int(match.group(1))
        inches = int(match.group(2))
        pounds = float(match.group(3))
        
        # Convert to standard metric units
        total_inches = (feet * 12) + inches
        height_cm = total_inches * 2.54
        weight_kg = pounds * 0.453592
        
        return pd.Series([height_cm, weight_kg])
    else:
        # If the row is empty or formatted weirdly, return None
        return pd.Series([None, None])

print("Parsing combined Height & Weight column...")

# 3. Apply the function to the specific column from your image
# Make sure the column name exactly matches your CSV (e.g., 'Height & Weight')
df[['height_cm', 'weight_kg']] = df['Height & Weight'].apply(parse_and_convert)

# 4. Clean up the dataset
# Drop rows where the parsing failed or data was missing
clean_df = df.dropna(subset=['height_cm', 'weight_kg'])

# (Optional) Drop the old messy column if you don't need it anymore
# clean_df = clean_df.drop(columns=['Height & Weight'])

# 5. Save the final clean version
clean_df.to_csv('clean_dataset.csv', index=False)
print("✅ Dataset parsed, converted, and cleaned! Saved as clean_dataset.csv")