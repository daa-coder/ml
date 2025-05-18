# Import libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Sample transactional data
dataset = [
    ['Milk', 'Bread', 'Butter'],
    ['Beer', 'Bread'],
    ['Milk', 'Diaper', 'Beer', 'Eggs'],
    ['Bread', 'Butter', 'Diaper'],
    ['Milk', 'Bread', 'Butter', 'Beer'],
    ['Bread', 'Diaper', 'Eggs'],
]

# Convert transactional data into one-hot encoded DataFrame
all_items = sorted(set(item for transaction in dataset for item in transaction))
encoded_vals = []
for transaction in dataset:
    row = {item: (item in transaction) for item in all_items}
    encoded_vals.append(row)
df = pd.DataFrame(encoded_vals)

# Apply Apriori to find frequent itemsets with minimum support of 0.5
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

# Generate association rules with minimum confidence of 0.7
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Display frequent itemsets and rules
print("Frequent Itemsets:\n", frequent_itemsets)
print("\nAssociation Rules:\n", rules)
