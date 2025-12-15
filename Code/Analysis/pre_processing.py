import pandas as pd
import re

df = pd.read_csv('final.csv')

print(f"Initial rows: {len(df)}")

def clean_text_for_nlp(text):
    if pd.isna(text):
        return ""
    
    text = str(text)
    text = text.lower()
    #Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    #Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    #Keep only English letters and spaces (remove numbers, special chars)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    #Remove extra whitespace
    text = ' '.join(text.split())
    
    #Remove single character words (optional, but often useful)
    text = ' '.join([word for word in text.split() if len(word) > 1])
    
    return text

df['like_text'] = df['like_text'].apply(clean_text_for_nlp)
df['dislike_text'] = df['dislike_text'].apply(clean_text_for_nlp)

print(f"After cleaning text: {len(df)}")

MIN_LENGTH = 2  

df = df[
    (df['like_text'].str.len() >= MIN_LENGTH) & 
    (df['dislike_text'].str.len() >= MIN_LENGTH)
]

print(f"After removing short reviews: {len(df)}")

df['review_year'] = df['review_date'].str.extract(r'(\d{4})')[0]

cols = df.columns.tolist()
date_idx = cols.index('review_date')
cols.insert(date_idx + 1, cols.pop(cols.index('review_year')))
df = df[cols]

print(f"Added review_year column")

# Remove companies with less than 10 reviews
company_counts = df['company'].value_counts()
companies_to_keep = company_counts[company_counts >= 10].index
df = df[df['company'].isin(companies_to_keep)]

print(f"After removing companies with < 10 reviews: {len(df)} rows, {df['company'].nunique()} companies")

df.to_csv('final_cleaned.csv', index=False)

print(f"Cleaned data saved to 'final_cleaned.csv'")
