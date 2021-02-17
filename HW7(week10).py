import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sb

#dataset
df = pd.DataFrame({'house size':[3529,3247,4032,2397,2200,3536,2983],
                    'lot size':[9191,10061,10150,14156,9600,19994,9365],
                    'bedrooms':[6,5,5,4,4,6,5],
                    'granite':[0,1,0,1,0,1,0],
                  'upgraded bathroom':[0,1,1,0,1,1,1],
                  'selling price':[205000,224900,197900,189900,195000,325000,230000]})
x = df['house size']
y = df['selling price']


# k=2
# initializes the data frame as a numpy object
points = df.values
# perform k-means algorithm based on data
kmeans = KMeans(n_clusters=2).fit(points)
# find the central position of each cluster
kmeans.cluster_centers_
# identify the cluster to which each data belongs
kmeans.labels_

# record data per cluster for visualization
df['cluster'] = kmeans.labels_
# outputs as a result of the final clustering completion
sb.lmplot('house size', 'selling price', data = df, fit_reg=False, scatter_kws={"s":50},hue="cluster")
plt.title('K-means Clustering, k=2')



# k=3
# initializes the data frame as a numpy object
points=df.values
# perform k-means algorithm based on data
kmeans=KMeans(n_clusters=3).fit(points)
# find the central position of each cluster
kmeans.cluster_centers_
# identify the cluster to which each data belongs
kmeans.labels_

# record data per cluster for visualization
df['cluster']=kmeans.labels_
# outputs as a result of the final clustering completion
sb.lmplot('house size', 'selling price', data=df,fit_reg=False, scatter_kws={"s":50},hue="cluster")
plt.title('K-means Clustering, k=3')



######################################################################################
# Without Scaling
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler

# data set
df = pd.DataFrame({'house size': [3529, 3247, 4032, 2397, 2200, 3536, 2983],
                   'lot size': [9191, 10061, 10150, 14156, 9600, 19994, 9365],
                   'bedrooms': [6, 5, 5, 4, 4, 6, 5],
                   'granite': [0, 1, 0, 1, 0, 1, 0],
                   'upgraded bathroom': [0, 1, 1, 0, 1, 1, 1],
                   'selling price': [205000, 224900, 197900, 189900, 195000, 325000, 230000]})

x = np.array(df.drop(['upgraded bathroom'], 1))
y = np.array(df['upgraded bathroom'])

# perform k-means algorithm based on data
kmeans = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
                n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
                random_state=None, tol=0.0001, verbose=0)

# create model
kmeans.fit(x)

# calculating predicted probabilities
# the machine finds the cluster and assigns any value to the cluster in the order it finds it
# a group of upgraded bathroom can be 0 or 1
correct = 0
for i in range(len(x)):
    predict_me = np.array(x[i])
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print("The result of prdiction probability:", correct / len(x))



################################################################################
# With Scaling (Parameter Changes)
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler

# data set
df = pd.DataFrame({'house size': [3529, 3247, 4032, 2397, 2200, 3536, 2983],
                   'lot size': [9191, 10061, 10150, 14156, 9600, 19994, 9365],
                   'bedrooms': [6, 5, 5, 4, 4, 6, 5],
                   'granite': [0, 1, 0, 1, 0, 1, 0],
                   'upgraded bathroom': [0, 1, 1, 0, 1, 1, 1],
                   'selling price': [205000, 224900, 197900, 189900, 195000, 325000, 230000]})

x = np.array(df.drop(['upgraded bathroom'], 1))
y = np.array(df['upgraded bathroom'])

# perform k-means algorithm based on data
kmeans = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=900,
                n_clusters=2, n_init=5, n_jobs=1, precompute_distances='auto',
                random_state=None, tol=0.0001, verbose=0)

# create model
kmeans.fit(x)

# calculating predicted probabilities
# the machine finds the cluster and assigns any value to the cluster in the order it finds it
# a group of upgraded bathroom can be 0 or 1
correct = 0
for i in range(len(x)):
    predict_me = np.array(x[i])
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print("The result of prdiction probability:", correct / len(x))



#########################################################################
# Without Scaling
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler

# data set
df = pd.DataFrame({'house size': [3529, 3247, 4032, 2397, 2200, 3536, 2983],
                   'lot size': [9191, 10061, 10150, 14156, 9600, 19994, 9365],
                   'bedrooms': [6, 5, 5, 4, 4, 6, 5],
                   'granite': [0, 1, 0, 1, 0, 1, 0],
                   'upgraded bathroom': [0, 1, 1, 0, 1, 1, 1],
                   'selling price': [205000, 224900, 197900, 189900, 195000, 325000, 230000]})

x = np.array(df.drop(['upgraded bathroom'], 1))
y = np.array(df['upgraded bathroom'])

# scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(x)

# perform k-means algorithm based on data
kmeans = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
                n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
                random_state=None, tol=0.0001, verbose=0)

# create model
kmeans.fit(X_scaled)

# calculating predicted probabilities
# the machine finds the cluster and assigns any value to the cluster in the order it finds it
# a group of upgraded bathroom can be 0 or 1
correct = 0
for i in range(len(X_scaled)):
    predict_me = np.array(X_scaled[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print("The result of prdiction probability:", correct / len(X_scaled))


######################################################################################3
# With Scaling(Parameter Changes)
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler

# data set
df = pd.DataFrame({'house size': [3529, 3247, 4032, 2397, 2200, 3536, 2983],
                   'lot size': [9191, 10061, 10150, 14156, 9600, 19994, 9365],
                   'bedrooms': [6, 5, 5, 4, 4, 6, 5],
                   'granite': [0, 1, 0, 1, 0, 1, 0],
                   'upgraded bathroom': [0, 1, 1, 0, 1, 1, 1],
                   'selling price': [205000, 224900, 197900, 189900, 195000, 325000, 230000]})

x = np.array(df.drop(['upgraded bathroom'], 1))
y = np.array(df['upgraded bathroom'])

# scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(x)

# perform k-means algorithm based on data
kmeans = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=900,
                n_clusters=2, n_init=5, n_jobs=1, precompute_distances='auto',
                random_state=None, tol=0.0001, verbose=0)

# create model
kmeans.fit(X_scaled)

# calculating predicted probabilities
# the machine finds the cluster and assigns any value to the cluster in the order it finds it
# a group of upgraded bathroom can be 0 or 1
correct = 0
for i in range(len(X_scaled)):
    predict_me = np.array(X_scaled[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print("The result of prdiction probability:", correct / len(X_scaled))


##################################################################################3
#Linkage
import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# data set
df=pd.DataFrame({'1':[0,9,3,6,11],
                 '2':[9,0,7,5,10],
                 '3':[3,7,0,9,2],
                 '4':[6,5,9,0,8],
                 '5':[11,10,2,8,0]})
#know the distance
distance=squareform(df)

#dengrogram 'complete linkage'
dendrogram(
    linkage(distance, method='complete'), labels=[1,2,3,4,5])
plt.title('Complete linkage')
plt.show()

#dengrogram 'average linkage'
dendrogram(
    linkage(distance, method='average'), labels=[1,2,3,4,5])
plt.title('Average linkage')
plt.show()

#dengrogram 'centroid linkage'
dendrogram(
    linkage(distance, method='centroid'), labels=[1,2,3,4,5])
plt.title('Centroid linkage')
plt.show()
