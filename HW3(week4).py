import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

# MinMax Scaler
np.random.seed(1)
df=pd.DataFrame({
    'x1':np.random.normal(0,2,10000),
    'x2':np.random.normal(5,3,10000),
    'x3':np.random.normal(-5,5,10000)
})
np.random.normal(loc=0.0,scale=1.0,size=None)
scaler=preprocessing.MinMaxScaler()
scaled_df=scaler.fit_transform(df)
scaled_df=pd.DataFrame(scaled_df,columns=['x1','x2','x3'])

fig,(ax1,ax2)=plt.subplots(ncols=2,figsize=(6,5))

ax1.set_title('Before Scaling')
sns.kdeplot(df['x1'],ax=ax1)
sns.kdeplot(df['x2'],ax=ax1)
sns.kdeplot(df['x3'],ax=ax1)

ax2.set_title('After MinMax Scaler')
sns.kdeplot(scaled_df['x1'],ax=ax2)
sns.kdeplot(scaled_df['x2'],ax=ax2)
sns.kdeplot(scaled_df['x3'],ax=ax2)
plt.show()


# Robust Scaler
np.random.seed(1)
df=pd.DataFrame({
    'x1':np.random.normal(0,2,10000),
    'x2':np.random.normal(5,3,10000),
    'x3':np.random.normal(-5,5,10000)
})
np.random.normal(loc=0.0,scale=1.0,size=None)
scaler=preprocessing.RobustScaler()
scaled_df=scaler.fit_transform(df)
scaled_df=pd.DataFrame(scaled_df,columns=['x1','x2','x3'])

fig,(ax1,ax2)=plt.subplots(ncols=2,figsize=(6,5))

ax1.set_title('Before Scaling')
sns.kdeplot(df['x1'],ax=ax1)
sns.kdeplot(df['x2'],ax=ax1)
sns.kdeplot(df['x3'],ax=ax1)

ax2.set_title('After Robust Scaler')
sns.kdeplot(scaled_df['x1'],ax=ax2)
sns.kdeplot(scaled_df['x2'],ax=ax2)
sns.kdeplot(scaled_df['x3'],ax=ax2)
plt.show()


#Standard Scaling-1
data = np.array([20,15,26,32,18,28,35,14,26,22,17])

print("The mean:", data.mean())
print("The standard deviation:", "%.1f"%data.std())

standard_scores = (data-data.mean())/data.std()
result = np.around(standard_scores,2)

for i in range(result.size):
    if result[i] < -1:
        print(data[i], end=' ')


# Standard Scaling-2
data = np.array([28, 35, 26, 32, 28, 28, 35, 34, 46, 42, 37])

print("The mean:", "%.1f" % data.mean())
print("The standard deviation:", "%.1f" % data.std())

standard_scores = (data-data.mean())/data.std()
result = np.around(standard_scores, 2)

for i in range(result.size):
    if result[i] < -1:
        print(data[i], end=' ')
        
