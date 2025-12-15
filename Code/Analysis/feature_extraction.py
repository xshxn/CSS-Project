import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

df = pd.read_csv('Data/Final/final_cleaned.csv')

print(f"Initial shape: {df.shape}")

# Contraction mapping
contractions_dict = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "couldn't": "could not", "won't": "will not",
    "wouldn't": "would not", "shouldn't": "should not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
    "it's": "it is", "we're": "we are", "they're": "they are",
    "i've": "i have", "you've": "you have", "we've": "we have",
    "they've": "they have", "i'll": "i will", "you'll": "you will",
    "he'll": "he will", "she'll": "she will", "we'll": "we will",
    "they'll": "they will", "i'd": "i would", "you'd": "you would",
    "he'd": "he would", "she'd": "she would", "we'd": "we would",
    "they'd": "they would"
}

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def expand_contractions(text):
    if pd.isna(text) or text == "":
        return ""
    text_lower = str(text).lower()
    for contraction, expansion in contractions_dict.items():
        text_lower = text_lower.replace(contraction, expansion)
    return text_lower

def advanced_preprocess(text):
    if pd.isna(text) or text == "":
        return ""
    # Expand contractions
    text = expand_contractions(text)
    # Remove stop words and lemmatize
    words = text.split()
    processed_words = [
        lemmatizer.lemmatize(word) 
        for word in words 
        if word not in stop_words and len(word) > 1
    ]
    
    return ' '.join(processed_words)

print("Preprocessing text (contractions, stop words, lemmatization)...")
df['like_text_processed'] = df['like_text'].apply(advanced_preprocess)
df['dislike_text_processed'] = df['dislike_text'].apply(advanced_preprocess)

print("preprocessing complete")

def get_sentiment_features(text):
    if pd.isna(text) or text == "":
        return 0.0, 0.0
    blob = TextBlob(str(text))
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# Apply sentiment analysis on PROCESSED text
df[['like_sentiment_polarity', 'like_sentiment_subjectivity']] = df['like_text_processed'].apply(
    lambda x: pd.Series(get_sentiment_features(x))
)
df[['dislike_sentiment_polarity', 'dislike_sentiment_subjectivity']] = df['dislike_text_processed'].apply(
    lambda x: pd.Series(get_sentiment_features(x))
)

# Combined sentiment (difference between likes and dislikes)
df['sentiment_difference'] = df['like_sentiment_polarity'] - df['dislike_sentiment_polarity']

print("Sentiment features extracted")

df['like_text_length'] = df['like_text_processed'].astype(str).apply(len)
df['dislike_text_length'] = df['dislike_text_processed'].astype(str).apply(len)
df['like_word_count'] = df['like_text_processed'].astype(str).apply(lambda x: len(x.split()))
df['dislike_word_count'] = df['dislike_text_processed'].astype(str).apply(lambda x: len(x.split()))
df['text_length_ratio'] = df['like_text_length'] / (df['dislike_text_length'] + 1)  # Avoid division by zero

print("Text length features extracted")
financial_keywords = {
    'salary': ['salary', 'pay', 'compensation', 'bonus', 'increment', 'hike', 'package'],
    'growth': ['growth', 'learning', 'skill', 'training', 'development', 'career', 'promotion'],
    'culture': ['culture', 'environment', 'atmosphere', 'team', 'management', 'leadership'],
    'worklife': ['work life balance', 'worklife', 'timing', 'hours', 'flexible', 'leaves', 'holiday'],
    'security': ['security', 'stable', 'permanent', 'job security', 'layoff', 'fired'],
    'stress': ['stress', 'pressure', 'workload', 'overwork', 'burden', 'exhausted'],
    'negative': ['bad', 'worst', 'poor', 'terrible', 'horrible', 'pathetic', 'useless']
}

def count_keyword_mentions(text, keywords):
    if pd.isna(text) or text == "":
        return 0
    text_lower = str(text).lower()
    count = sum(1 for keyword in keywords if keyword in text_lower)
    return count

for category, keywords in financial_keywords.items():
    df[f'like_{category}_mentions'] = df['like_text_processed'].apply(lambda x: count_keyword_mentions(x, keywords))
    df[f'dislike_{category}_mentions'] = df['dislike_text_processed'].apply(lambda x: count_keyword_mentions(x, keywords))

print("✓ Keyword frequency features extracted")

# Group by company and calculate aggregate statistics
company_features = df.groupby('company').agg({
    'overall_rating': ['mean', 'std', 'count'],
    'like_sentiment_polarity': ['mean', 'std'],
    'dislike_sentiment_polarity': ['mean', 'std'],
    'sentiment_difference': 'mean',
    'like_word_count': 'mean',
    'dislike_word_count': 'mean',
}).reset_index()

company_features.columns = ['_'.join(col).strip('_') for col in company_features.columns.values]
df = df.merge(company_features, on='company', how='left', suffixes=('', '_company_avg'))

print("Company-level aggregate features created")

df['combined_text_processed'] = df['like_text_processed'].astype(str) + " " + df['dislike_text_processed'].astype(str)

tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df['combined_text_processed'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])])

df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)

print("TF-IDF features extracted")
if 'review_year' in df.columns:
    df['review_year'] = pd.to_numeric(df['review_year'], errors='coerce')
    year_counts = df.groupby(['company', 'review_year']).size().reset_index(name='reviews_per_year')
    df = df.merge(year_counts, on=['company', 'review_year'], how='left')
    print("Temporal features extracted")
df.to_csv('Data/Final/data_with_features.csv', index=False)

print(f"Feature extraction complete!")
print(f"Final shape: {df.shape}")
print(f"Features extracted: {df.shape[1]} columns")
print(f"\nKey feature categories:")
print(f"  - Sentiment: 5 features")
print(f"  - Text length: 5 features")
print(f"  - Keywords: {len(financial_keywords) * 2} features")
print(f"  - Company aggregates: {len(company_features.columns) - 1} features")
print(f"  - TF-IDF: 100 features")
if 'review_year' in df.columns:
    print(f"  - Temporal: 1 feature")

