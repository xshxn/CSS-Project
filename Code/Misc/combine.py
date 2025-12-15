import pandas as pd

df1 = pd.read_csv('ambitionbox_likes_dislikes_with_overall.csv')
df2 = pd.read_csv('ambitionbox_likes_dislikes_with_overall_2.csv')
df3 = pd.read_csv('ambitionbox_likes_dislikes_with_overall_uday.csv')
df4 = pd.read_csv('ambitionbox_likes_dislikes_with_overall_otherhalf_shreyansh.csv')
df5 = pd.read_csv('ambitionbox_likes_dislikes_with_overall_shreyansh.csv')
df6 = pd.read_csv('ambitionbox_likes_dislikes_with_overall_new2_sam.csv')
df7 = pd.read_csv('ambitionbox_likes_dislikes_with_overall_new_sam.csv')
df8 = pd.read_csv('ambitionbox_likes_dislikes_with_overall_new3_sam.csv')

df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)

df['overall_rating'] = pd.to_numeric(df['overall_rating'], errors='coerce')

df = df[df['overall_rating'].notna() & (df['overall_rating'] % 1 == 0)]

# Remove rows where review_date is None, "NONE", or empty
df = df[df['review_date'].notna() & (df['review_date'] != 'NONE') & (df['review_date'].astype(str).str.strip() != '')]

# Remove duplicate rows
df = df.drop_duplicates()

df.to_csv('final.csv', index=False)

