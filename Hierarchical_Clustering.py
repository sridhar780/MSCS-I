import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

# Load the dataset
file_path = r"C:\Users\rbdee\OneDrive\Documents\Desktop\Telco_customer_churn.csv"
df = pd.read_csv(file_path)

# Drop unnecessary columns
df = df.drop(df.columns[[0, 1, 2]], axis=1)

df.info()

# Convert categorical columns into dummy variables
df = pd.get_dummies(df, dtype = int)

# Normalize the data (Min-Max normalization)
df_norm = (df - df.min()) / (df.max() - df.min())
df_norm.describe()
# Perform hierarchical clustering
linkage_matrix = linkage(df_norm, method="ward", metric="euclidean")

# Plot the Dendrogram
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()

# Create clusters (example: cutting the dendrogram at 3 clusters)
clusters = fcluster(linkage_matrix, t=3, criterion="maxclust")

# Add cluster labels to the original data
# Assign cluster labels to the original dataset
clus_labels = pd.DataFrame(clusters, columns=['cluster'])

df_final = pd.concat([clus_labels, df], axis=1)

# Display the first few rows with cluster labels
print("Data with Cluster Labels:")
print(df_final.head())

# Analyze clusters
clust_mean = df_final.groupby('cluster').mean()

# Save the clustered data to a CSV file
df_final.to_csv("Simple_Hierarchical_Clustering.csv", index=False)
print("\nClustered data saved to 'Simple_Hierarchical_Clustering.csv'")
from sklearn import metrics
#from clusteval import clusteval
#metrics.silhouette_score(df, clus_labels)
