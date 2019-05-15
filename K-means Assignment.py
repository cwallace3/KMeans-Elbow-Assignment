import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets

#Importing the Iris dataset

IrisData = datasets.load_iris()
df = pd.DataFrame(IrisData.data, columns = IrisData.feature_names)
IrisTarget = pd.DataFrame(IrisData.target, columns=['Target'])

#Finding the optimum number of clusters for k-means classification
from sklearn.cluster import KMeans
wcss = []

df = pd.concat([df, IrisTarget], join='inner', axis = 1)
x = df.iloc[:, [1, 2, 3, 4]].values


for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, 
                    n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


#Plotting the results on a line graph to observe the "elbow"    
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.savefig("ElbowMethod_Iris.png", dpi = 100)
plt.show()