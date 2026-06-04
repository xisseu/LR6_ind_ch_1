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