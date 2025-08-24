

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Load the dataset
df = pd.read_csv("customer_segmentationn/Mall_Customers.csv")

# Explore the data
print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Description:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Encode categorical variables (Gender)
le = LabelEncoder()
df['Gender_Encoded'] = le.fit_transform(df['Gender'])
print("\nGender encoding:")
for i, category in enumerate(le.classes_):
    print(f"{category}: {i}")

# Visualize distributions
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution')

plt.subplot(2, 2, 2)
sns.histplot(df['Annual Income (k$)'], kde=True)
plt.title('Annual Income Distribution')

plt.subplot(2, 2, 3)
sns.histplot(df['Spending Score (1-100)'], kde=True)
plt.title('Spending Score Distribution')

plt.subplot(2, 2, 4)
sns.countplot(x='Gender', data=df)
plt.title('Gender Distribution')

plt.tight_layout()
plt.show()

# Correlation analysis (using only numeric columns)
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Select relevant features for clustering
# We'll use Annual Income and Spending Score for 2D visualization
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
wcss = []  # Within-Cluster Sum of Squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the elbow method
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Using the Silhouette Score
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score Method')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Based on the elbow method and silhouette score, choose the optimal number of clusters
optimal_clusters = 5  # This is typically the optimal number for this dataset

# Apply K-Means
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original dataframe
df['Cluster'] = y_kmeans

# Visualize the clusters
plt.figure(figsize=(15, 7))

# Before clustering
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], s=50, alpha=0.7)
plt.title('Before Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')

# After clustering
plt.subplot(1, 2, 2)
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(optimal_clusters):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=50, c=colors[i], label=f'Cluster {i}', alpha=0.7)

# Plot the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='yellow', label='Centroids', marker='X')
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()

plt.tight_layout()
plt.show()

# Using Dendrogram to find the optimal number of clusters
plt.figure(figsize=(15, 7))
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Apply Hierarchical Clustering (updated)
hc = AgglomerativeClustering(n_clusters=optimal_clusters, linkage='ward')
y_hc = hc.fit_predict(X_scaled)

# Add hierarchical cluster labels to the dataframe
df['HC_Cluster'] = y_hc

# Visualize hierarchical clusters
plt.figure(figsize=(10, 7))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']  # Make sure colors is defined
for i in range(optimal_clusters):
    plt.scatter(X[y_hc == i, 0], X[y_hc == i, 1], s=50, c=colors[i], label=f'Cluster {i}', alpha=0.7)
plt.title('Hierarchical Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Analyze cluster characteristics
cluster_analysis = df.groupby('Cluster').agg({
    'Age': 'mean',
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean',
    'Gender': lambda x: x.value_counts().index[0]  # Most common gender in cluster
}).round(2)

print("Cluster Analysis:")
print(cluster_analysis)

# Create customer profiles based on clusters
profiles = {
    0: "Standard Customers (Moderate Income, Moderate Spending)",
    1: "Careful Customers (High Income, Low Spending)",
    2: "Target Customers (High Income, High Spending)",
    3: "Sensible Customers (Low Income, Low Spending)",
    4: "Careless Customers (Low Income, High Spending)"
}

df['Segment'] = df['Cluster'].map(profiles)

# Visualize segment distribution
plt.figure(figsize=(12, 6))
sns.countplot(x='Segment', data=df)
plt.title('Customer Segment Distribution')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 3D Visualization with more features (optional)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Use three features for 3D visualization
x = df['Age']
y = df['Annual Income (k$)']
z = df['Spending Score (1-100)']

ax.scatter(x, y, z, c=df['Cluster'], cmap='viridis', s=50, alpha=0.6)
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
ax.set_title('3D Cluster Visualization')
plt.show()

# Business Insights
print("\nBusiness Insights:")
print("1. Target Customers (Cluster 2): High income and high spending - These are your most valuable customers.")
print("2. Careless Customers (Cluster 4): Low income but high spending - They might need financial advice or could be at risk of debt.")
print("3. Careful Customers (Cluster 1): High income but low spending - They might need more persuasion or better offers.")
print("4. Standard Customers (Cluster 0): Moderate income and spending - The largest group, good for general marketing.")
print("5. Sensible Customers (Cluster 3): Low income and low spending - Might be price-sensitive, respond well to discounts.")

# Save the results
df.to_csv('customer_segmentationn/customer_segmentation_results.csv', index=False)
print("\nResults saved to 'customer_segmentation_results.csv'")