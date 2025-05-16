import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score

# Sample DataFrame
data = {
    'query': ['brake pad', 'brake pad', 'brake pad', 'engine coil', 'engine coil', 'engine coil'],
    'part_number': ['P1001', 'P1002', 'P1003', 'P2001', 'P2002', 'P2003'],
    'cart_adds': [10, 5, 0, 3, 7, 0],
    'predicted_score': [0.92, 0.85, 0.80, 0.88, 0.83, 0.75]
}

df = pd.DataFrame(data)

# Compute NDCG per query
ndcg_scores = []
for query, group in df.groupby('query'):
    true_relevance = np.array(group['cart_adds']).reshape(1, -1)
    predicted_scores = np.array(group['predicted_score']).reshape(1, -1)
    
    score = ndcg_score(true_relevance, predicted_scores)
    ndcg_scores.append(score)
    print(f"NDCG for '{query}': {score:.4f}")

# Average NDCG across queries
avg_ndcg = np.mean(ndcg_scores)
print(f"\nAverage NDCG: {avg_ndcg:.4f}")
