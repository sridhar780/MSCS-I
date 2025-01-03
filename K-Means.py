import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
file_path = r"C:\Users\rbdee\OneDrive\Documents\Desktop\Telco_customer_churn.csv"
df = pd.read_csv(file_path)

# Drop unnecessary columns
df = df.drop(df.columns[[0, 1, 2]], axis=1)  # Dropping columns by index - 0th and 2nd column

df.info()

# Convert categorical columns to dummy variables
df = pd.get_dummies(df, columns=['Payment Method', 'Contract', 'Offer', 'Internet Type',
                                 'Referred a Friend', 'Phone Service', 'Multiple Lines',
                                 'Internet Service', 'Online Security', 'Online Backup',
                                 'Device Protection Plan', 'Premium Tech Support',
                                 'Streaming TV', 'Streaming Movies', 'Streaming Music',
                                 'Unlimited Data', 'Paperless Billing'], dtype = int)

# Normalization function
def normalize(data):
    return (data - data.min()) / (data.max() - data.min())

# Normalize the data
df_norm = normalize(df)

eda = df_norm.describe()

# Elbow Method - Find the optimal number of clusters
inertia = []
clusters = list(range(2, 10))  # Trying k values from 2 to 9

for k in clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_norm)
    inertia.append(kmeans.inertia_)

# Plot the Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(clusters, inertia, 'bo-')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
plt.title("Elbow Method for Optimal k")
plt.show()

# Apply K-Means Clustering with k=3 (based on the Elbow curve result)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df_norm)

# Assign cluster labels to the original dataset
clus_labels = pd.DataFrame(kmeans.labels_, columns=['cluster'])

df_final = pd.concat([clus_labels, df], axis=1)

# Display first few rows with cluster labels
print("Data with Cluster Labels:")
print(df_final.head())

# Cluster Analysis - Mean values for each cluster
clust_mean = df_final.groupby('cluster').mean()


# Save the clustered data to a CSV file
output_path = "KMeans_Telco_Customer_Clusters.csv"
df_final.to_csv(output_path, index=False)
print(f"\nClustered data saved to {output_path}")

import os
os.getcwd()
