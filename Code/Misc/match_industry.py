import pandas as pd

# Read the data files
df_cleaned = pd.read_csv('final_cleaned.csv')
df_industry = pd.read_csv('data_industry.csv')

print(f"Data cleaned: {len(df_cleaned)} rows")
print(f"Data industry: {len(df_industry)} rows")

industry_lookup = {}

for _, row in df_industry.iterrows():
    company = str(row['company']).lower().strip()
    industry = row['industry']
    if company and company != 'nan':
        industry_lookup[company] = industry

for _, row in df_industry.iterrows():
    company_formal = str(row['company_formal']).lower().strip()
    industry = row['industry']
    if company_formal and company_formal != 'nan':
        industry_lookup[company_formal] = industry

print(f"Industry lookup dictionary has {len(industry_lookup)} entries")

def get_industry(company_name):
    if pd.isna(company_name):
        return None
    company_lower = str(company_name).lower().strip()
    return industry_lookup.get(company_lower, None)

df_cleaned['industry'] = df_cleaned['company'].apply(get_industry)

matched = df_cleaned['industry'].notna().sum()
total = len(df_cleaned)
print(f"\nMatching Results:")
print(f"  Matched: {matched} ({matched/total*100:.1f}%)")
print(f"  Unmatched: {total - matched} ({(total-matched)/total*100:.1f}%)")

unmatched_companies = df_cleaned[df_cleaned['industry'].isna()]['company'].unique()
if len(unmatched_companies) > 0:
    print(f"\nSample unmatched companies (first 10):")
    for company in unmatched_companies[:10]:
        print(f"  - {company}")

df_cleaned.to_csv('final_cleaned_industry.csv', index=False)
print(f"\nUpdated final_cleaned.csv with industry column")
