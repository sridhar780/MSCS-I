import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load the dataset
file_path = r"E:\Downloads\retail_data.xlsx"
df = pd.read_excel(file_path)

# Preview the dataset
print("Dataset Preview:")
print(df.head())

# Step 1: Prepare the data for Market Basket Analysis
# Assuming 'BillNo' represents transactions and 'Itemname' represents items
basket = df.groupby(['BillNo', 'Itemname'])['Quantity'].sum().unstack().fillna(0)

# Convert positive quantities to 1 (binary format for transactions)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

print("\nPrepared Basket Data (First 5 rows):")
print(basket.head())

# Step 2: Generate frequent itemsets
frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True, verbose=1)

print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Step 3: Generate association rules
try:
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    print("\nAssociation Rules:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
except TypeError as e:
    print("\nError:", e)
    print("Ensure that mlxtend version 0.21.0 or later is installed.")

# Save the association rules to a CSV file
output_path = "Apriori_Association_Rules.csv"
rules.to_csv(output_path, index=False)
print(f"\nAssociation rules saved to '{output_path}'")
