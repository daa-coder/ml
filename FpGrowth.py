# Import libraries
pip install mlxtend
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

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
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply FP-Growth to find frequent itemsets with minimum support of 0.5
frequent_itemsets = fpgrowth(df, min_support=0.5, use_colnames=True)

# Generate association rules with minimum confidence of 0.7
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Display frequent itemsets and association rules
print("Frequent Itemsets:\n", frequent_itemsets)
print("\nAssociation Rules:\n", rules)
