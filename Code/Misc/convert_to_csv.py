import pandas as pd

df = pd.read_excel('data.xlsx', header=None)

print(f"Number of columns: {len(df.columns)}")
print(f"First few rows:\n{df.head()}")

column_names = ['company', 'company_formal', 'column3', 'column4', 'column5', 'column6', 'column7', 'column8', 'column9', 'column10', 'column11', 'column12', 'column13', 'column14', 'column15', 'column16', 'column17', 'industry', 'column18', 'column19', 'column20', 'column21', 'column22', 'column23', 'column24', 'column25', 'column26', 'column27', 'column28', 'column29', 'column30', 'column31', 'column32', 'column33', 'column34', 'column35', 'column36', 'column37', 'column38', 'column39', 'column40', 'column41', 'column42', 'column43', 'column44']  # Replace with your actual column names
df.columns = column_names

df.to_csv('data_industry.csv', index=False)
print("Saved to data.csv")