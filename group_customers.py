import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#load the dataset
data = pd.read_csv("D:\git demo\RtlStore\Mall_Customers.csv")

#select the relevant features
features = data[['Age', 'Annual Income (k$)' , 'Spending Score (1-100)']]

#normalise the data
scalar = StandardScaler()
scaled_features = scalar.fit_transform(features)

#find the optimal cluster using Elbow method
inertia = []
for i in range(1,201):
    kmeans = KMeans(n_clusters = i, random_state = 0)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

plt.plot(range(1,201), inertia, marker = 'o')
plt.xlabel('Number of clusters')
plt.ylabel('inertia')
plt.title('elbow method')
plt.show()


# Apply K-Means with the chosen number of clusters
optimal_clusters = 13  # Replace with the number of clusters from the Elbow Method
kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
clusters = kmeans.fit_predict(scaled_features)

# Add cluster labels to the original data
data['Cluster'] = clusters

# Analyze clusters
#cluster_summary = data.groupby('Cluster').mean()
#print(cluster_summary)

# Group by 'Cluster' and calculate the mean only for numeric columns
numeric_columns = data.select_dtypes(include='number').columns
cluster_summary = data.groupby('Cluster')[numeric_columns].mean()

print(cluster_summary)


# Optionally, visualize the clusters
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Amount spent')
plt.title('Customer Clustering')
plt.colorbar(label='Cluster')
plt.show()





