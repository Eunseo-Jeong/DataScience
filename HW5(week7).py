from pandas import DataFrame as df
import math
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

#1
df1 = df(data={'Anxiety(X)':[10,8,2,1,5,6],
             'Test score(Y)':[2,3,9,7,6,5],
            'X2':[100,64,4,1,25,36],
            'Y2':[4,9,81,49,36,25],
            'XY':[20,24,18,7,30,30]})
df1.loc[6,'Anxiety(X)']=df1['Anxiety(X)'].sum()
df1.loc[6,'Test score(Y)']=df1['Test score(Y)'].sum()
df1.loc[6,'X2']= df1['X2'].sum()
df1.loc[6,'Y2']=df1['Y2'].sum()
df1.loc[6,'XY']=df1['XY'].sum()

df1 = df1.rename(index={6:'Total'})

result1 = df1.iloc[6]['XY']-(df1.iloc[6]['Anxiety(X)']*df1.iloc[6]['Test score(Y)'])/(len(df1)-1)
result2 = math.sqrt((df1.iloc[6]['X2']-(df1.iloc[6]['Anxiety(X)']**2)/(len(df1)-1))*(df1.iloc[6]['Y2']-(df1.iloc[6]['Test score(Y)']**2)/(len(df1)-1)))
r = result1/result2
print('r= %.2f'%r)


#2
data = {'Person':[1,2,3,4,5],
     'Age':[30,40,50,60,40],
     'Income':[200,300,800,600,300],
     'Yrs worked':[10,20,20,20,20],
     'Vacation':[4,4,1,2,5]}

df = pd.DataFrame(data,columns=['Person','Age','Income','Yrs worked','Vacation'])
covMatrix = pd.DataFrame.cov(df)
print(covMatrix)

sn.heatmap(covMatrix,annot=True,fmt='g')
plt.show()
