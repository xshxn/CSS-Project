import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('final.csv')

total_unique_companies = df['company'].nunique()
print(f"Total unique companies: {total_unique_companies}")

avg_reviews_per_company = df.groupby('company').size().mean()
print(f"Average number of reviews per company: {avg_reviews_per_company:.2f}")

# Distribution of overall ratings
print("\nDistribution of Overall Ratings:")
rating_dist = df['overall_rating'].value_counts().sort_index()
print(rating_dist)
print(f"\nPercentage distribution:")
print((rating_dist / len(df) * 100).round(2))

# Average rating across all reviews
avg_rating = df['overall_rating'].mean()
print(f"\nAverage overall rating across all reviews: {avg_rating:.2f}")

plt.figure(figsize=(10, 6))
rating_dist.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribution of Overall Ratings', fontsize=14, fontweight='bold')
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Number of Reviews', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('rating_distribution.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved rating distribution plot as 'rating_distribution.png'")

df['like_length'] = df['like_text'].apply(lambda x: len(str(x)) if pd.notna(x) and str(x) != 'NONE' else 0)
df['dislike_length'] = df['dislike_text'].apply(lambda x: len(str(x)) if pd.notna(x) and str(x) != 'NONE' else 0)

avg_like_length = df['like_length'].mean()
avg_dislike_length = df['dislike_length'].mean()

print(f"\nAverage length of likes text: {avg_like_length:.2f} characters")
print(f"Average length of dislikes text: {avg_dislike_length:.2f} characters")

df['total_text_length'] = df['like_length'] + df['dislike_length']
company_avg_length = df.groupby('company')['total_text_length'].mean().sort_values()

# Company with shortest average text
shortest_company = company_avg_length.idxmin()
shortest_length = company_avg_length.min()

# Company with longest average text
longest_company = company_avg_length.idxmax()
longest_length = company_avg_length.max()

print(f"\nCompany with SHORTEST average review text:")
print(f"  {shortest_company}: {shortest_length:.2f} characters")

print(f"\nCompany with LONGEST average review text:")
print(f"  {longest_company}: {longest_length:.2f} characters")

print("TOP 5 COMPANIES BY AVG TEXT LENGTH")
print("\nLongest:")
print(company_avg_length.tail(5))
print("\nShortest:")
print(company_avg_length.head(5))