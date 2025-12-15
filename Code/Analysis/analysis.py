import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

import os
output_dir = 'analysis_plots'
os.makedirs(output_dir, exist_ok=True)

print("Loading data...")
df = pd.read_csv('final_cleaned.csv')
print(f"Total reviews loaded: {len(df)}")
print(f"Total unique companies: {df['company'].nunique()}")


fig, ax = plt.subplots(figsize=(12, 6))
rating_counts = df['overall_rating'].value_counts().sort_index()

# Create bar plot with gradient colors
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(rating_counts)))
bars = ax.bar(rating_counts.index.astype(str), rating_counts.values, color=colors, edgecolor='black', linewidth=0.5)

# Add value labels on bars
for bar, val in zip(bars, rating_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
            f'{val:,}\n({val/len(df)*100:.1f}%)', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Overall Rating', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Reviews', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Overall Ratings Across All Companies', fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0, max(rating_counts.values) * 1.15)

# Add summary statistics
mean_rating = df['overall_rating'].mean()
median_rating = df['overall_rating'].median()
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
textstr = f'Mean: {mean_rating:.2f}\nMedian: {median_rating:.1f}\nTotal: {len(df):,}'
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{output_dir}/01_rating_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/01_rating_distribution.png")
print(f"  Mean Rating: {mean_rating:.2f}")
print(f"  Median Rating: {median_rating:.1f}")


# Calculate text lengths
df['like_length'] = df['like_text'].fillna('').astype(str).apply(len)
df['dislike_length'] = df['dislike_text'].fillna('').astype(str).apply(len)
df['like_word_count'] = df['like_text'].fillna('').astype(str).apply(lambda x: len(x.split()))
df['dislike_word_count'] = df['dislike_text'].fillna('').astype(str).apply(lambda x: len(x.split()))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Character length comparison
length_data = pd.DataFrame({
    'Type': ['Likes'] * len(df) + ['Dislikes'] * len(df),
    'Character Length': list(df['like_length']) + list(df['dislike_length'])
})

# Filter outliers for better visualization (keep 99th percentile)
upper_limit = length_data['Character Length'].quantile(0.99)
length_data_filtered = length_data[length_data['Character Length'] <= upper_limit]

sns.boxplot(data=length_data_filtered, x='Type', y='Character Length', ax=axes[0], 
            palette=['#2ecc71', '#e74c3c'], width=0.5)
axes[0].set_title('Character Length Distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('')
axes[0].set_ylabel('Number of Characters', fontsize=10)

# Add mean annotations
like_mean = df['like_length'].mean()
dislike_mean = df['dislike_length'].mean()
axes[0].text(0, axes[0].get_ylim()[1] * 0.95, f'Mean: {like_mean:.0f}', ha='center', fontsize=9, fontweight='bold')
axes[0].text(1, axes[0].get_ylim()[1] * 0.95, f'Mean: {dislike_mean:.0f}', ha='center', fontsize=9, fontweight='bold')

# Word count comparison
word_data = pd.DataFrame({
    'Type': ['Likes'] * len(df) + ['Dislikes'] * len(df),
    'Word Count': list(df['like_word_count']) + list(df['dislike_word_count'])
})

upper_limit_words = word_data['Word Count'].quantile(0.99)
word_data_filtered = word_data[word_data['Word Count'] <= upper_limit_words]

sns.boxplot(data=word_data_filtered, x='Type', y='Word Count', ax=axes[1], 
            palette=['#2ecc71', '#e74c3c'], width=0.5)
axes[1].set_title('Word Count Distribution', fontsize=12, fontweight='bold')
axes[1].set_xlabel('')
axes[1].set_ylabel('Number of Words', fontsize=10)

# Add mean annotations
like_word_mean = df['like_word_count'].mean()
dislike_word_mean = df['dislike_word_count'].mean()
axes[1].text(0, axes[1].get_ylim()[1] * 0.95, f'Mean: {like_word_mean:.1f}', ha='center', fontsize=9, fontweight='bold')
axes[1].text(1, axes[1].get_ylim()[1] * 0.95, f'Mean: {dislike_word_mean:.1f}', ha='center', fontsize=9, fontweight='bold')

plt.suptitle('Comparison of Like vs Dislike Text Lengths (All Reviews)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/02_text_length_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/02_text_length_comparison.png")
print(f"  Average Like Length: {like_mean:.0f} chars, {like_word_mean:.1f} words")
print(f"  Average Dislike Length: {dislike_mean:.0f} chars, {dislike_word_mean:.1f} words")


# Combine all text
all_likes = ' '.join(df['like_text'].fillna('').astype(str).tolist())
all_dislikes = ' '.join(df['dislike_text'].fillna('').astype(str).tolist())

# Common stopwords to exclude (business-related generic terms)
stopwords = set(['based', 'user', 'ratings', 'none', 'nothing', 'good', 'no', 'yes', 'na', 
                 'nil', 'nill', 'company', 'work', 'working', 'job', 'employee', 'employees'])

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Likes Word Cloud
wc_likes = WordCloud(width=800, height=400, background_color='white', 
                     colormap='Greens', max_words=100, stopwords=stopwords,
                     min_font_size=10, max_font_size=150).generate(all_likes)
axes[0].imshow(wc_likes, interpolation='bilinear')
axes[0].axis('off')
axes[0].set_title('What Employees LIKE', fontsize=14, fontweight='bold', color='#27ae60', pad=10)

# Dislikes Word Cloud
wc_dislikes = WordCloud(width=800, height=400, background_color='white', 
                        colormap='Reds', max_words=100, stopwords=stopwords,
                        min_font_size=10, max_font_size=150).generate(all_dislikes)
axes[1].imshow(wc_dislikes, interpolation='bilinear')
axes[1].axis('off')
axes[1].set_title('What Employees DISLIKE', fontsize=14, fontweight='bold', color='#c0392b', pad=10)

plt.suptitle('Word Cloud Analysis of Employee Reviews', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(f'{output_dir}/03_word_clouds.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/03_word_clouds.png")


df['review_year'] = pd.to_numeric(df['review_year'], errors='coerce')

df_temporal = df[(df['review_year'] >= 2015) & (df['review_year'] <= 2025)].copy()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

yearly_counts = df_temporal.groupby('review_year').size()
axes[0, 0].bar(yearly_counts.index, yearly_counts.values, color='#3498db', edgecolor='black', linewidth=0.5)
axes[0, 0].set_xlabel('Year', fontsize=10)
axes[0, 0].set_ylabel('Number of Reviews', fontsize=10)
axes[0, 0].set_title('Reviews per Year', fontsize=12, fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=45)
for i, (year, count) in enumerate(zip(yearly_counts.index, yearly_counts.values)):
    axes[0, 0].text(year, count + 500, f'{count:,}', ha='center', fontsize=8)

yearly_rating = df_temporal.groupby('review_year')['overall_rating'].mean()
axes[0, 1].plot(yearly_rating.index, yearly_rating.values, marker='o', linewidth=2, 
                markersize=8, color='#e74c3c', markerfacecolor='white', markeredgewidth=2)
axes[0, 1].fill_between(yearly_rating.index, yearly_rating.values, alpha=0.3, color='#e74c3c')
axes[0, 1].set_xlabel('Year', fontsize=10)
axes[0, 1].set_ylabel('Average Rating', fontsize=10)
axes[0, 1].set_title('Average Rating Trend Over Years', fontsize=12, fontweight='bold')
axes[0, 1].set_ylim(1, 5)
axes[0, 1].axhline(y=yearly_rating.mean(), color='gray', linestyle='--', alpha=0.7, label=f'Overall Mean: {yearly_rating.mean():.2f}')
axes[0, 1].legend()

rating_by_year = df_temporal.groupby(['review_year', 'overall_rating']).size().unstack(fill_value=0)
rating_by_year_pct = rating_by_year.div(rating_by_year.sum(axis=1), axis=0) * 100
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(rating_by_year_pct.columns)))
rating_by_year_pct.plot(kind='bar', stacked=True, ax=axes[1, 0], color=colors, edgecolor='white', linewidth=0.5)
axes[1, 0].set_xlabel('Year', fontsize=10)
axes[1, 0].set_ylabel('Percentage (%)', fontsize=10)
axes[1, 0].set_title('Rating Distribution by Year (%)', fontsize=12, fontweight='bold')
axes[1, 0].legend(title='Rating', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
axes[1, 0].tick_params(axis='x', rotation=45)

yearly_lengths = df_temporal.groupby('review_year').agg({
    'like_word_count': 'mean',
    'dislike_word_count': 'mean'
}).rename(columns={'like_word_count': 'Likes', 'dislike_word_count': 'Dislikes'})

yearly_lengths.plot(ax=axes[1, 1], marker='o', linewidth=2, markersize=6)
axes[1, 1].set_xlabel('Year', fontsize=10)
axes[1, 1].set_ylabel('Average Word Count', fontsize=10)
axes[1, 1].set_title('Average Review Length Trend', fontsize=12, fontweight='bold')
axes[1, 1].legend()

plt.suptitle('Temporal Analysis of Employee Reviews', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/04_temporal_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/04_temporal_analysis.png")


likes_corpus = df['like_text'].fillna('').astype(str).tolist()
dislikes_corpus = df['dislike_text'].fillna('').astype(str).tolist()

custom_stopwords = ['based', 'user', 'ratings', 'none', 'nothing', 'good', 'no', 'yes', 'na', 
                    'nil', 'nill', 'company', 'work', 'working', 'job', 'employee', 'employees',
                    'like', 'dislike', 'very', 'also', 'get', 'one', 'would', 'could', 'much',
                    'really', 'even', 'well', 'lot', 'many', 'every', 'everything', 'anything']

# TF-IDF for Likes
tfidf_likes = TfidfVectorizer(max_features=50, stop_words='english', ngram_range=(1, 2), 
                               min_df=100, max_df=0.8)
tfidf_likes_matrix = tfidf_likes.fit_transform(likes_corpus)
likes_feature_names = tfidf_likes.get_feature_names_out()
likes_scores = np.asarray(tfidf_likes_matrix.mean(axis=0)).flatten()
likes_tfidf_df = pd.DataFrame({'word': likes_feature_names, 'tfidf': likes_scores}).sort_values('tfidf', ascending=False)

# TF-IDF for Dislikes
tfidf_dislikes = TfidfVectorizer(max_features=50, stop_words='english', ngram_range=(1, 2), 
                                  min_df=100, max_df=0.8)
tfidf_dislikes_matrix = tfidf_dislikes.fit_transform(dislikes_corpus)
dislikes_feature_names = tfidf_dislikes.get_feature_names_out()
dislikes_scores = np.asarray(tfidf_dislikes_matrix.mean(axis=0)).flatten()
dislikes_tfidf_df = pd.DataFrame({'word': dislikes_feature_names, 'tfidf': dislikes_scores}).sort_values('tfidf', ascending=False)

# Find unique words (in likes but not in dislikes, and vice versa)
likes_words = set(likes_tfidf_df['word'].tolist())
dislikes_words = set(dislikes_tfidf_df['word'].tolist())
unique_to_likes = likes_words - dislikes_words
unique_to_dislikes = dislikes_words - likes_words

print(f"  Top unique words in LIKES (not in dislikes): {len(unique_to_likes)}")
print(f"  Top unique words in DISLIKES (not in likes): {len(unique_to_dislikes)}")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Top 20 Likes terms
top_likes = likes_tfidf_df.head(20)
axes[0].barh(top_likes['word'], top_likes['tfidf'], color='#27ae60', edgecolor='black', linewidth=0.5)
axes[0].set_xlabel('TF-IDF Score', fontsize=10)
axes[0].set_title('Top 20 Terms in LIKES (by TF-IDF)', fontsize=12, fontweight='bold', color='#27ae60')
axes[0].invert_yaxis()

# Top 20 Dislikes terms
top_dislikes = dislikes_tfidf_df.head(20)
axes[1].barh(top_dislikes['word'], top_dislikes['tfidf'], color='#c0392b', edgecolor='black', linewidth=0.5)
axes[1].set_xlabel('TF-IDF Score', fontsize=10)
axes[1].set_title('Top 20 Terms in DISLIKES (by TF-IDF)', fontsize=12, fontweight='bold', color='#c0392b')
axes[1].invert_yaxis()

plt.suptitle('TF-IDF Analysis: Distinctive Terms in Likes vs Dislikes', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/05_tfidf_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/05_tfidf_analysis.png")


fig, axes = plt.subplots(1, 2, figsize=(16, 7))

likes_word_freq = dict(zip(likes_tfidf_df['word'], likes_tfidf_df['tfidf'] * 1000))
wc_likes_tfidf = WordCloud(width=800, height=400, background_color='white', 
                            colormap='Greens', max_words=50,
                            min_font_size=12, max_font_size=100).generate_from_frequencies(likes_word_freq)
axes[0].imshow(wc_likes_tfidf, interpolation='bilinear')
axes[0].axis('off')
axes[0].set_title('Distinctive Words in LIKES\n(TF-IDF Weighted)', fontsize=12, fontweight='bold', color='#27ae60')

# Create word frequency dict from TF-IDF scores for dislikes
dislikes_word_freq = dict(zip(dislikes_tfidf_df['word'], dislikes_tfidf_df['tfidf'] * 1000))
wc_dislikes_tfidf = WordCloud(width=800, height=400, background_color='white', 
                               colormap='Reds', max_words=50,
                               min_font_size=12, max_font_size=100).generate_from_frequencies(dislikes_word_freq)
axes[1].imshow(wc_dislikes_tfidf, interpolation='bilinear')
axes[1].axis('off')
axes[1].set_title('Distinctive Words in DISLIKES\n(TF-IDF Weighted)', fontsize=12, fontweight='bold', color='#c0392b')

plt.suptitle('TF-IDF Weighted Word Clouds: Most Distinguishing Terms', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(f'{output_dir}/06_tfidf_wordclouds.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/06_tfidf_wordclouds.png")



print(f"""
Dataset Overview:
  - Total Reviews: {len(df):,}
  - Unique Companies: {df['company'].nunique():,}
  - Date Range: {int(df_temporal['review_year'].min())} - {int(df_temporal['review_year'].max())}

Rating Statistics:
  - Mean Rating: {df['overall_rating'].mean():.2f}
  - Median Rating: {df['overall_rating'].median():.1f}
  - Mode Rating: {df['overall_rating'].mode().values[0]}

Text Length Statistics:
  - Avg Like Length: {df['like_length'].mean():.0f} characters ({df['like_word_count'].mean():.1f} words)
  - Avg Dislike Length: {df['dislike_length'].mean():.0f} characters ({df['dislike_word_count'].mean():.1f} words)

All plots saved to: {output_dir}/
  1. 01_rating_distribution.png
  2. 02_text_length_comparison.png
  3. 03_word_clouds.png
  4. 04_temporal_analysis.png
  5. 05_tfidf_analysis.png
  6. 06_tfidf_wordclouds.png
""")

# Save summary to CSV
summary_data = {
    'Metric': ['Total Reviews', 'Unique Companies', 'Mean Rating', 'Median Rating', 
               'Avg Like Length (chars)', 'Avg Dislike Length (chars)',
               'Avg Like Words', 'Avg Dislike Words'],
    'Value': [len(df), df['company'].nunique(), round(df['overall_rating'].mean(), 2),
              df['overall_rating'].median(), round(df['like_length'].mean(), 0),
              round(df['dislike_length'].mean(), 0), round(df['like_word_count'].mean(), 1),
              round(df['dislike_word_count'].mean(), 1)]
}
pd.DataFrame(summary_data).to_csv(f'{output_dir}/analysis_summary.csv', index=False)
print(f"Summary statistics saved to: {output_dir}/analysis_summary.csv")
