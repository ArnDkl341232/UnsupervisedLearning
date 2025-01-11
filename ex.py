import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans



data = {
    'Medium amount of purchases per month' : [10, 4, 33, 12, 8, 9, 57, 5, 6, 28],
    'Medium check' : [200, 100, 50, 400, 640, 230, 110, 990, 360, 120],
    'Arequency of visits' : [4, 19, 16, 9, 7, 4, 5, 20, 14, 17],
}

df = pd.DataFrame(data)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Claster'] = kmeans.fit_predict(df)

centroids = kmeans.cluster_centers_

plt.scatter(df['Medium amount of purchases per month'], df['Medium check'], c=df['Claster'], cmap='viridis', s=50)
plt.scatter(centroids[:,0], centroids[:,1], c='red',s=200, label="Centroid" , marker="x")
plt.title("Result clusters")
plt.xlabel("Medium amount of purchases per month")
plt.ylabel("Medium check")
plt.colorbar(label='Claster')
plt.show()