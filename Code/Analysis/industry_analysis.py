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
output_dir = 'industry_analysis_plots'
os.makedirs(output_dir, exist_ok=True)

print("Loading data...")
df = pd.read_csv('final_cleaned_industry.csv')

df_with_industry = df[df['industry'].notna()].copy()
print(f"Total reviews: {len(df)}")
print(f"Reviews with industry info: {len(df_with_industry)}")
print(f"Unique industries: {df_with_industry['industry'].nunique()}")

df_with_industry['like_length'] = df_with_industry['like_text'].fillna('').astype(str).apply(len)
df_with_industry['dislike_length'] = df_with_industry['dislike_text'].fillna('').astype(str).apply(len)
df_with_industry['like_word_count'] = df_with_industry['like_text'].fillna('').astype(str).apply(lambda x: len(x.split()))
df_with_industry['dislike_word_count'] = df_with_industry['dislike_text'].fillna('').astype(str).apply(lambda x: len(x.split()))

# Get top industries by review count for focused analysis
top_industries = df_with_industry['industry'].value_counts().head(10).index.tolist()
df_top = df_with_industry[df_with_industry['industry'].isin(top_industries)].copy()
print(f"\nTop 10 industries (by review count):")
for i, ind in enumerate(top_industries, 1):
    count = len(df_with_industry[df_with_industry['industry'] == ind])
    print(f"  {i}. {ind}: {count:,} reviews")


fig, ax = plt.subplots(figsize=(14, 8))
industry_ratings = df_top.groupby('industry')['overall_rating'].mean().sort_values(ascending=True)

colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(industry_ratings)))
bars = ax.barh(industry_ratings.index, industry_ratings.values, color=colors, edgecolor='black', linewidth=0.5)

# Add value labels
for bar, val in zip(bars, industry_ratings.values):
    ax.text(val + 0.05, bar.get_y() + bar.get_height()/2, f'{val:.2f}', 
            va='center', fontsize=9, fontweight='bold')

ax.set_xlabel('Average Overall Rating', fontsize=12, fontweight='bold')
ax.set_ylabel('Industry', fontsize=12, fontweight='bold')
ax.set_title('Average Overall Rating by Industry (Top 10 Industries)', fontsize=14, fontweight='bold')
ax.set_xlim(0, 5)
ax.axvline(x=df_top['overall_rating'].mean(), color='red', linestyle='--', 
           label=f'Overall Mean: {df_top["overall_rating"].mean():.2f}', linewidth=2)
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig(f'{output_dir}/01_rating_by_industry.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/01_rating_by_industry.png")


fig, axes = plt.subplots(1, 2, figsize=(18, 10))

# Likes word count by industry
likes_data = df_top[['industry', 'like_word_count']].copy()
likes_data = likes_data[likes_data['like_word_count'] <= likes_data['like_word_count'].quantile(0.95)]

order = df_top.groupby('industry')['like_word_count'].median().sort_values(ascending=False).index
sns.boxplot(data=likes_data, y='industry', x='like_word_count', ax=axes[0], 
            order=order, palette='Greens_r', width=0.6)
axes[0].set_xlabel('Word Count', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Industry', fontsize=11, fontweight='bold')
axes[0].set_title('Distribution of LIKES Word Count by Industry', fontsize=12, fontweight='bold', color='#27ae60')

# Dislikes word count by industry
dislikes_data = df_top[['industry', 'dislike_word_count']].copy()
dislikes_data = dislikes_data[dislikes_data['dislike_word_count'] <= dislikes_data['dislike_word_count'].quantile(0.95)]

order = df_top.groupby('industry')['dislike_word_count'].median().sort_values(ascending=False).index
sns.boxplot(data=dislikes_data, y='industry', x='dislike_word_count', ax=axes[1], 
            order=order, palette='Reds_r', width=0.6)
axes[1].set_xlabel('Word Count', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Industry', fontsize=11, fontweight='bold')
axes[1].set_title('Distribution of DISLIKES Word Count by Industry', fontsize=12, fontweight='bold', color='#c0392b')

plt.suptitle('Text Length Analysis by Industry', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/02_text_length_by_industry.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/02_text_length_by_industry.png")


# Calculate industry averages
industry_avg = df_with_industry.groupby('industry')['overall_rating'].mean().to_dict()

# Calculate company ratings and compare
company_stats = df_with_industry.groupby(['industry', 'company']).agg({
    'overall_rating': ['mean', 'count']
}).reset_index()
company_stats.columns = ['industry', 'company', 'company_avg', 'review_count']
company_stats['industry_avg'] = company_stats['industry'].map(industry_avg)
company_stats['diff_from_industry'] = company_stats['company_avg'] - company_stats['industry_avg']

# For each top industry, show companies above/below average
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, industry in enumerate(top_industries[:6]):
    industry_data = company_stats[company_stats['industry'] == industry].copy()
    industry_data = industry_data[industry_data['review_count'] >= 50]  # At least 50 reviews
    
    if len(industry_data) < 5:
        axes[idx].text(0.5, 0.5, f'Not enough data\nfor {industry[:20]}...', 
                      ha='center', va='center', transform=axes[idx].transAxes)
        axes[idx].set_title(f'{industry[:30]}...', fontsize=10, fontweight='bold')
        continue
    
    # Get top 5 above and below average
    top_above = industry_data.nlargest(5, 'diff_from_industry')
    top_below = industry_data.nsmallest(5, 'diff_from_industry')
    plot_data = pd.concat([top_above, top_below]).drop_duplicates()
    plot_data = plot_data.sort_values('diff_from_industry')
    
    colors = ['#e74c3c' if x < 0 else '#27ae60' for x in plot_data['diff_from_industry']]
    bars = axes[idx].barh(plot_data['company'].str[:20], plot_data['diff_from_industry'], color=colors)
    axes[idx].axvline(x=0, color='black', linewidth=1)
    axes[idx].set_xlabel('Diff from Industry Avg', fontsize=9)
    axes[idx].set_title(f'{industry[:35]}', fontsize=10, fontweight='bold')

plt.suptitle('Company Ratings vs Industry Average\n(Green = Above Avg, Red = Below Avg)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/03_company_vs_industry_avg.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/03_company_vs_industry_avg.png")


fig, axes = plt.subplots(1, 3, figsize=(18, 8))

# Total reviews by industry
review_counts = df_top.groupby('industry').size().sort_values(ascending=True)
axes[0].barh(review_counts.index, review_counts.values, color='#3498db', edgecolor='black')
axes[0].set_xlabel('Number of Reviews', fontsize=11)
axes[0].set_title('Total Reviews by Industry', fontsize=12, fontweight='bold')

# Average likes/dislikes length comparison
industry_lengths = df_top.groupby('industry').agg({
    'like_word_count': 'mean',
    'dislike_word_count': 'mean'
}).sort_values('like_word_count', ascending=True)

x = np.arange(len(industry_lengths))
width = 0.35
axes[1].barh(x - width/2, industry_lengths['like_word_count'], width, label='Likes', color='#27ae60')
axes[1].barh(x + width/2, industry_lengths['dislike_word_count'], width, label='Dislikes', color='#e74c3c')
axes[1].set_yticks(x)
axes[1].set_yticklabels(industry_lengths.index)
axes[1].set_xlabel('Average Word Count', fontsize=11)
axes[1].set_title('Avg Likes vs Dislikes Length by Industry', fontsize=12, fontweight='bold')
axes[1].legend()

# Ratio of likes to dislikes length
industry_lengths['ratio'] = industry_lengths['like_word_count'] / industry_lengths['dislike_word_count']
industry_lengths_sorted = industry_lengths.sort_values('ratio', ascending=True)
colors = ['#27ae60' if x > 1 else '#e74c3c' for x in industry_lengths_sorted['ratio']]
axes[2].barh(industry_lengths_sorted.index, industry_lengths_sorted['ratio'], color=colors)
axes[2].axvline(x=1, color='black', linestyle='--', linewidth=2)
axes[2].set_xlabel('Ratio (Likes/Dislikes Length)', fontsize=11)
axes[2].set_title('Likes:Dislikes Length Ratio by Industry', fontsize=12, fontweight='bold')

plt.suptitle('Likes vs Dislikes Analysis by Industry', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/04_likes_dislikes_by_industry.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/04_likes_dislikes_by_industry.png")


stopwords = set(['based', 'user', 'ratings', 'none', 'nothing', 'good', 'no', 'yes', 'na', 
                 'nil', 'nill', 'company', 'work', 'working', 'job', 'employee', 'employees',
                 'like', 'dislike', 'very', 'also', 'get', 'one', 'would', 'could', 'much'])

fig, axes = plt.subplots(4, 2, figsize=(16, 20))

for idx, industry in enumerate(top_industries[:4]):
    industry_df = df_with_industry[df_with_industry['industry'] == industry]
    
    # Likes word cloud
    all_likes = ' '.join(industry_df['like_text'].fillna('').astype(str).tolist())
    if len(all_likes.strip()) > 100:
        wc_likes = WordCloud(width=600, height=300, background_color='white', 
                            colormap='Greens', max_words=60, stopwords=stopwords).generate(all_likes)
        axes[idx, 0].imshow(wc_likes, interpolation='bilinear')
    axes[idx, 0].axis('off')
    axes[idx, 0].set_title(f'LIKES: {industry[:40]}', fontsize=10, fontweight='bold', color='#27ae60')
    
    # Dislikes word cloud
    all_dislikes = ' '.join(industry_df['dislike_text'].fillna('').astype(str).tolist())
    if len(all_dislikes.strip()) > 100:
        wc_dislikes = WordCloud(width=600, height=300, background_color='white', 
                               colormap='Reds', max_words=60, stopwords=stopwords).generate(all_dislikes)
        axes[idx, 1].imshow(wc_dislikes, interpolation='bilinear')
    axes[idx, 1].axis('off')
    axes[idx, 1].set_title(f'DISLIKES: {industry[:40]}', fontsize=10, fontweight='bold', color='#c0392b')

plt.suptitle('Word Clouds by Industry: What Employees Say', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{output_dir}/05_wordclouds_by_industry.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/05_wordclouds_by_industry.png")


df_with_industry['review_year'] = pd.to_numeric(df_with_industry['review_year'], errors='coerce')
df_temporal = df_with_industry[(df_with_industry['review_year'] >= 2018) & 
                                (df_with_industry['review_year'] <= 2025)].copy()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

yearly_by_industry = df_temporal.groupby(['review_year', 'industry']).size().unstack(fill_value=0)
yearly_by_industry[top_industries[:5]].plot(ax=axes[0, 0], marker='o', linewidth=2)
axes[0, 0].set_xlabel('Year', fontsize=11)
axes[0, 0].set_ylabel('Number of Reviews', fontsize=11)
axes[0, 0].set_title('Review Volume Trend by Industry (Top 5)', fontsize=12, fontweight='bold')
axes[0, 0].legend(loc='upper left', fontsize=8)

yearly_rating_by_industry = df_temporal.groupby(['review_year', 'industry'])['overall_rating'].mean().unstack()
yearly_rating_by_industry[top_industries[:5]].plot(ax=axes[0, 1], marker='o', linewidth=2)
axes[0, 1].set_xlabel('Year', fontsize=11)
axes[0, 1].set_ylabel('Average Rating', fontsize=11)
axes[0, 1].set_title('Rating Trend by Industry (Top 5)', fontsize=12, fontweight='bold')
axes[0, 1].set_ylim(1, 5)
axes[0, 1].legend(loc='upper left', fontsize=8)

yearly_length = df_temporal.groupby(['review_year', 'industry'])['like_word_count'].mean().unstack()
yearly_length[top_industries[:5]].plot(ax=axes[1, 0], marker='o', linewidth=2)
axes[1, 0].set_xlabel('Year', fontsize=11)
axes[1, 0].set_ylabel('Avg Like Word Count', fontsize=11)
axes[1, 0].set_title('Like Review Length Trend by Industry (Top 5)', fontsize=12, fontweight='bold')
axes[1, 0].legend(loc='upper left', fontsize=8)

yearly_dislike_length = df_temporal.groupby(['review_year', 'industry'])['dislike_word_count'].mean().unstack()
yearly_dislike_length[top_industries[:5]].plot(ax=axes[1, 1], marker='o', linewidth=2)
axes[1, 1].set_xlabel('Year', fontsize=11)
axes[1, 1].set_ylabel('Avg Dislike Word Count', fontsize=11)
axes[1, 1].set_title('Dislike Review Length Trend by Industry (Top 5)', fontsize=12, fontweight='bold')
axes[1, 1].legend(loc='upper left', fontsize=8)

plt.suptitle('Temporal Analysis by Industry', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/06_temporal_by_industry.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/06_temporal_by_industry.png")


fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, industry in enumerate(top_industries[:6]):
    industry_df = df_with_industry[df_with_industry['industry'] == industry]
    
    # Get top 2 companies by review count with at least 100 reviews
    company_counts = industry_df.groupby('company').size()
    top_companies = company_counts[company_counts >= 100].nlargest(2).index.tolist()
    
    if len(top_companies) < 2:
        top_companies = company_counts.nlargest(2).index.tolist()
    
    if len(top_companies) >= 2:
        compare_data = []
        for company in top_companies:
            company_df = industry_df[industry_df['company'] == company]
            compare_data.append({
                'Company': company[:20],
                'Likes Avg Words': company_df['like_word_count'].mean(),
                'Dislikes Avg Words': company_df['dislike_word_count'].mean(),
                'Avg Rating': company_df['overall_rating'].mean()
            })
        compare_df = pd.DataFrame(compare_data)
        
        x = np.arange(len(compare_df))
        width = 0.35
        axes[idx].bar(x - width/2, compare_df['Likes Avg Words'], width, label='Likes', color='#27ae60')
        axes[idx].bar(x + width/2, compare_df['Dislikes Avg Words'], width, label='Dislikes', color='#e74c3c')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(compare_df['Company'], rotation=15, ha='right')
        
        # Add rating annotation
        for i, (_, row) in enumerate(compare_df.iterrows()):
            axes[idx].annotate(f"★{row['Avg Rating']:.1f}", (i, max(row['Likes Avg Words'], row['Dislikes Avg Words']) + 2),
                              ha='center', fontsize=9, fontweight='bold')
        
        axes[idx].set_ylabel('Avg Word Count', fontsize=10)
        axes[idx].set_title(f'{industry[:35]}', fontsize=11, fontweight='bold')
        axes[idx].legend(loc='upper right', fontsize=8)

plt.suptitle('Comparing Top Companies within Each Industry\n(Likes vs Dislikes Review Length)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/07_company_comparison_by_industry.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/07_company_comparison_by_industry.png")


# Analyze top 4 industries
fig, axes = plt.subplots(4, 2, figsize=(18, 20))

for idx, industry in enumerate(top_industries[:4]):
    industry_df = df_with_industry[df_with_industry['industry'] == industry]
    
    likes_corpus = industry_df['like_text'].fillna('').astype(str).tolist()
    dislikes_corpus = industry_df['dislike_text'].fillna('').astype(str).tolist()
    
    # TF-IDF for Likes
    try:
        tfidf_likes = TfidfVectorizer(max_features=20, stop_words='english', ngram_range=(1, 2), 
                                       min_df=20, max_df=0.8)
        tfidf_likes_matrix = tfidf_likes.fit_transform(likes_corpus)
        likes_features = tfidf_likes.get_feature_names_out()
        likes_scores = np.asarray(tfidf_likes_matrix.mean(axis=0)).flatten()
        likes_tfidf_df = pd.DataFrame({'word': likes_features, 'tfidf': likes_scores}).nlargest(15, 'tfidf')
        
        axes[idx, 0].barh(likes_tfidf_df['word'], likes_tfidf_df['tfidf'], color='#27ae60')
        axes[idx, 0].invert_yaxis()
    except:
        axes[idx, 0].text(0.5, 0.5, 'Not enough data', ha='center', va='center', transform=axes[idx, 0].transAxes)
    axes[idx, 0].set_xlabel('TF-IDF Score', fontsize=9)
    axes[idx, 0].set_title(f'LIKES: {industry[:35]}', fontsize=10, fontweight='bold', color='#27ae60')
    
    # TF-IDF for Dislikes
    try:
        tfidf_dislikes = TfidfVectorizer(max_features=20, stop_words='english', ngram_range=(1, 2), 
                                          min_df=20, max_df=0.8)
        tfidf_dislikes_matrix = tfidf_dislikes.fit_transform(dislikes_corpus)
        dislikes_features = tfidf_dislikes.get_feature_names_out()
        dislikes_scores = np.asarray(tfidf_dislikes_matrix.mean(axis=0)).flatten()
        dislikes_tfidf_df = pd.DataFrame({'word': dislikes_features, 'tfidf': dislikes_scores}).nlargest(15, 'tfidf')
        
        axes[idx, 1].barh(dislikes_tfidf_df['word'], dislikes_tfidf_df['tfidf'], color='#e74c3c')
        axes[idx, 1].invert_yaxis()
    except:
        axes[idx, 1].text(0.5, 0.5, 'Not enough data', ha='center', va='center', transform=axes[idx, 1].transAxes)
    axes[idx, 1].set_xlabel('TF-IDF Score', fontsize=9)
    axes[idx, 1].set_title(f'DISLIKES: {industry[:35]}', fontsize=10, fontweight='bold', color='#c0392b')

plt.suptitle('TF-IDF Analysis: Distinctive Terms by Industry', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{output_dir}/08_tfidf_by_industry.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/08_tfidf_by_industry.png")


fig, axes = plt.subplots(4, 2, figsize=(16, 20))

for idx, industry in enumerate(top_industries[:4]):
    industry_df = df_with_industry[df_with_industry['industry'] == industry]
    
    likes_corpus = industry_df['like_text'].fillna('').astype(str).tolist()
    dislikes_corpus = industry_df['dislike_text'].fillna('').astype(str).tolist()
    
    # TF-IDF Word Cloud for Likes
    try:
        tfidf_likes = TfidfVectorizer(max_features=50, stop_words='english', min_df=10, max_df=0.8)
        tfidf_likes_matrix = tfidf_likes.fit_transform(likes_corpus)
        likes_features = tfidf_likes.get_feature_names_out()
        likes_scores = np.asarray(tfidf_likes_matrix.mean(axis=0)).flatten()
        likes_freq = dict(zip(likes_features, likes_scores * 1000))
        
        wc = WordCloud(width=600, height=300, background_color='white', colormap='Greens').generate_from_frequencies(likes_freq)
        axes[idx, 0].imshow(wc, interpolation='bilinear')
    except:
        axes[idx, 0].text(0.5, 0.5, 'Not enough data', ha='center', va='center', transform=axes[idx, 0].transAxes)
    axes[idx, 0].axis('off')
    axes[idx, 0].set_title(f'LIKES (TF-IDF): {industry[:35]}', fontsize=10, fontweight='bold', color='#27ae60')
    
    # TF-IDF Word Cloud for Dislikes
    try:
        tfidf_dislikes = TfidfVectorizer(max_features=50, stop_words='english', min_df=10, max_df=0.8)
        tfidf_dislikes_matrix = tfidf_dislikes.fit_transform(dislikes_corpus)
        dislikes_features = tfidf_dislikes.get_feature_names_out()
        dislikes_scores = np.asarray(tfidf_dislikes_matrix.mean(axis=0)).flatten()
        dislikes_freq = dict(zip(dislikes_features, dislikes_scores * 1000))
        
        wc = WordCloud(width=600, height=300, background_color='white', colormap='Reds').generate_from_frequencies(dislikes_freq)
        axes[idx, 1].imshow(wc, interpolation='bilinear')
    except:
        axes[idx, 1].text(0.5, 0.5, 'Not enough data', ha='center', va='center', transform=axes[idx, 1].transAxes)
    axes[idx, 1].axis('off')
    axes[idx, 1].set_title(f'DISLIKES (TF-IDF): {industry[:35]}', fontsize=10, fontweight='bold', color='#c0392b')

plt.suptitle('TF-IDF Weighted Word Clouds by Industry', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{output_dir}/09_tfidf_wordclouds_by_industry.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/09_tfidf_wordclouds_by_industry.png")



print(f"""
All plots saved to: {output_dir}/
  1. 01_rating_by_industry.png
  2. 02_text_length_by_industry.png
  3. 03_company_vs_industry_avg.png
  4. 04_likes_dislikes_by_industry.png
  5. 05_wordclouds_by_industry.png
  6. 06_temporal_by_industry.png
  7. 07_company_comparison_by_industry.png
  8. 08_tfidf_by_industry.png
  9. 09_tfidf_wordclouds_by_industry.png
""")
