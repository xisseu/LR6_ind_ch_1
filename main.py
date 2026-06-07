#import matplotlib.pyplot as plt
#fig,ax= plt. subplots()
#ax.plot([1,2,3,4],[1,4,2,5])
#plt.ylabel('some numbers')
#plt.savefig('myfig.png')
from sklearn.datasets import make_blobs
import pandas as pd
dataset,classes = make_blobs(n_samples=200, n_features=2, centers =4,  cluster_std=0.5, random_state=0)
df = pd.DataFrame(dataset, columns=['var1','var2'])
print(df.head(2))
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,12)).fit(df)
visualizer.show(outpath="elbow_plot.png")
import matplotlib.pyplot as plt
plt.figure()

kmeans=KMeans(n_clusters=4,init='k-means++',random_state=0).fit(df)
print(kmeans.labels_)
print(kmeans.cluster_centers_)
print(kmeans.inertia_)
print(kmeans.n_iter_)
from collections import Counter
Counter(kmeans.labels_)
print(Counter(kmeans.labels_))

import seaborn as sns
sns.scatterplot(data=df, x='var1', y='var2', hue=kmeans.labels_)
plt.savefig('kmeans.png')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], marker="X", c="r", s=80, label="centroids")
plt.legend()
plt.savefig('kmeans_centroids.png')
