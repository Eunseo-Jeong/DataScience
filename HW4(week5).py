import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# warning
import warnings

warnings.filterwarnings(action='ignore')

# read excel file
df = pd.read_excel("bmi_data_phw3.xlsx")

# check is there any 'NaN' value
df.isnull().sum()

# print feature names
for i in range(len(df.columns)):
    print("Column {}: {}".format(i, df.columns[i]))

# print data types
print()
print(df.dtypes)

extremelyweak = 0
weak = 0
normal = 0
overweight = 0
obesity = 0
df0_h = []
df0_w = []
df1_h = []
df1_w = []
df2_h = []
df2_w = []
df3_h = []
df3_w = []
df4_h = []
df4_w = []

# BMI checking
for i in range(len(df)):
    if df['BMI'][i] == 0:
        extremelyweak += 1
        df0_h.append(df['Height (Inches)'][i])
        df0_w.append(df['Weight (Pounds)'][i])
    elif df['BMI'][i] == 1:
        weak += 1
        df1_h.append(df['Height (Inches)'][i])
        df1_w.append(df['Weight (Pounds)'][i])
    elif df['BMI'][i] == 2:
        normal += 1
        df2_h.append(df['Height (Inches)'][i])
        df2_w.append(df['Weight (Pounds)'][i])
    elif df['BMI'][i] == 3:
        overweight += 1
        df3_h.append(df['Height (Inches)'][i])
        df3_w.append(df['Weight (Pounds)'][i])
    elif df['BMI'][i] == 4:
        obesity += 1
        df4_h.append(df['Height (Inches)'][i])
        df4_w.append(df['Weight (Pounds)'][i])

print()
print('BMI_0: {}, BMI_1: {}, BMI_2: {}, BMI_3: {}, BMI_4: {}'
      .format(extremelyweak, weak, normal, overweight, obesity))

# Histogram BMI=0 height
plt.hist(df0_h, bins=10, rwidth=0.9)
plt.title('Histogram of normal(BMI=0)')
plt.xlabel('Height')
plt.ylabel('Number of Student')
plt.show()
# Histogram BMI=0 weight
plt.hist(df0_w, bins=10, rwidth=0.9)
plt.title('Histogram of normal(BMI=0)')
plt.xlabel('Weight')
plt.ylabel('Number of Student')
plt.show()

# Histogram BMI=1 height
plt.hist(df1_h, bins=10, rwidth=0.9)
plt.title('Histogram of normal(BMI=1)')
plt.xlabel('Height')
plt.ylabel('Number of Student')
plt.show()
# Histogram BMI=1 weight
plt.hist(df1_w, bins=10, rwidth=0.9)
plt.title('Histogram of normal(BMI=1)')
plt.xlabel('Weight')
plt.ylabel('Number of Student')
plt.show()

# Histogram BMI=2 height
plt.hist(df2_h, bins=10, rwidth=0.9)
plt.title('Histogram of normal(BMI=2)')
plt.xlabel('Height')
plt.ylabel('Number of Student')
plt.show()
# Histogram BMI=2 weight
plt.hist(df2_w, bins=10, rwidth=0.9)
plt.title('Histogram of normal(BMI=2)')
plt.xlabel('Weight')
plt.ylabel('Number of Student')
plt.show()

# Histogram BMI=3 height
plt.hist(df3_h, bins=10, rwidth=0.9)
plt.title('Histogram of normal(BMI=3)')
plt.xlabel('Height')
plt.ylabel('Number of Student')
plt.show()
# Histogram BMI=3 weight
plt.hist(df3_w, bins=10, rwidth=0.9)
plt.title('Histogram of normal(BMI=3)')
plt.xlabel('Weight')
plt.ylabel('Number of Student')
plt.show()

# Histogram BMI=4 height
plt.hist(df4_h, bins=10, rwidth=0.9)
plt.title('Histogram of normal(BMI=4)')
plt.xlabel('Height')
plt.ylabel('Number of Student')
plt.show()
# Histogram BMI=4 weight
plt.hist(df4_w, bins=10, rwidth=0.9)
plt.title('Histogram of normal(BMI=4)')
plt.xlabel('Weight')
plt.ylabel('Number of Student')
plt.show()

# Standard Scaling
df2 = df[df['BMI'] == 2]
df_2 = df2[['Height (Inches)', 'Weight (Pounds)']]
scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(df_2)
scaled_df = pd.DataFrame(scaled_df, columns=['Height', 'Weight'])
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))
ax1.set_title('Before Scaling')
sns.kdeplot(df_2['Height (Inches)'], ax=ax1)
sns.kdeplot(df_2['Weight (Pounds)'], ax=ax1)
ax2.set_title('After Standard Scaler')
sns.kdeplot(scaled_df['Height'], ax=ax2)
sns.kdeplot(scaled_df['Weight'], ax=ax2)
plt.show()

# Minmax Scaling
df2 = df[df['BMI'] == 2]
df_2 = df2[['Height (Inches)', 'Weight (Pounds)']]
scaler = preprocessing.MinMaxScaler()
scaled_df = scaler.fit_transform(df_2)
scaled_df = pd.DataFrame(scaled_df, columns=['Height', 'Weight'])
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))
ax1.set_title('Before Scaling')
sns.kdeplot(df_2['Height (Inches)'], ax=ax1)
sns.kdeplot(df_2['Weight (Pounds)'], ax=ax1)
ax2.set_title('After MinMax Scaler')
sns.kdeplot(scaled_df['Height'], ax=ax2)
sns.kdeplot(scaled_df['Weight'], ax=ax2)
plt.show()

# Robust Scaling
df2 = df[df['BMI'] == 2]
df_2 = df2[['Height (Inches)', 'Weight (Pounds)']]
scaler = preprocessing.RobustScaler()
scaled_df = scaler.fit_transform(df_2)
scaled_df = pd.DataFrame(scaled_df, columns=['Height', 'Weight'])
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))
ax1.set_title('Before Scaling')
sns.kdeplot(df_2['Height (Inches)'], ax=ax1)
sns.kdeplot(df_2['Weight (Pounds)'], ax=ax1)
ax2.set_title('After Robust Scaler')
sns.kdeplot(scaled_df['Height'], ax=ax2)
sns.kdeplot(scaled_df['Weight'], ax=ax2)
plt.show()

# Linear regression
X = df['Height (Inches)']
y = df['Weight (Pounds)']
line_fitter = LinearRegression()
line_fitter.fit(X.values.reshape(-1, 1), y)
line_fitter.coef_  # 기울기
line_fitter.intercept_  # y절편
plt.plot(X, y, 'o')
plt.plot(X, line_fitter.predict(X.values.reshape(-1, 1)))
plt.xlabel("Height (Inches)")
plt.ylabel("Weight (Pounds)")
plt.show()

# compute e
e = []
for i in range(len(df)):
    wexpect = df['Height (Inches)'][i]
    e.append(df['Weight (Pounds)'][i] - line_fitter.predict([[wexpect]]))
dfexp = pd.DataFrame(e)
dfexp.columns = ['e']

# compute Ze
ze = []
for i in range(len(df)):
    ze.append((dfexp['e'][i] - dfexp['e'].mean()) / dfexp['e'].std())
dfexp['Ze'] = pd.DataFrame(ze)
plt.hist(dfexp['Ze'].tolist(), bins=10, rwidth=0.9)
plt.xlabel('Ze')
plt.ylabel('frequency')
plt.show()

# Alpha correction
alpha = 1
correct = 0
count = 0
for i in range(len(df)):
    if dfexp['Ze'][i] < -alpha:
        count += 1
        if df['BMI'][i] == 0:
            correct += 1
            print("BMI 0-correct")
        else:
            print("BMI 0-not correct")
    elif dfexp['Ze'][i] > alpha:
        count += 1
        if df['BMI'][i] == 4:
            correct += 1
            print("BMI 4-correct")
        else:
            print("BMI 4-not correct")
print()
print('{}% correction'.format((correct / count) * 100))




#divide dataset into two groups
dfF=df[df['Sex']=='Female']
dfM=df[df['Sex']=='Male']


#Female linear regression
X=dfF['Height (Inches)']
y=dfF['Weight (Pounds)']
line_fitter=LinearRegression()
line_fitter.fit(X.values.reshape(-1,1),y)
line_fitter.coef_#기울기
line_fitter.intercept_#y절편
plt.plot(X,y,'o')
plt.plot(X,line_fitter.predict(X.values.reshape(-1,1)))
plt.title('Female linear regression')
plt.xlabel("Height (Inches)")
plt.ylabel("Weight (Pounds)")
plt.show()


#Female compute e
e=[]
for i in range(len(dfF)):
    wexpect=dfF['Height (Inches)'].iloc[i]
    e.append(dfF['Weight (Pounds)'].iloc[i]-line_fitter.predict([[wexpect]]))
dfFexp=pd.DataFrame(e)
dfFexp.columns=['e']


#Female compute Ze
ze=[]
for i in range(len(dfFexp)):
    ze.append((dfFexp['e'][i]-dfFexp['e'].mean())/dfFexp['e'].std())
dfFexp['Ze']=pd.DataFrame(ze)
plt.hist(dfFexp['Ze'].tolist(),bins=10,rwidth=0.9)
plt.title("Female Ze")
plt.xlabel('Ze')
plt.ylabel('frequency')
plt.show()

#Feale alpha correction
alpha=1
correct=0
count=0
for i in range(len(dfF)):
    if dfFexp['Ze'][i]<-alpha:
        count+=1
        if dfF['BMI'].iloc[i]==0:
            correct+=1
            print("BMI 0-correct")
        else:
            print("BMI 0-not correct")
    elif dfFexp['Ze'][i]>alpha:
        count+=1
        if dfF['BMI'].iloc[i]==4:
            correct+=1
            print("BMI 4-correct")
        else:
            print("BMI 4-not correct")

print()
print('{}% correction' .format((correct/count)*100))



#Male linear regression
X=dfM['Height (Inches)']
y=dfM['Weight (Pounds)']
line_fitter=LinearRegression()
line_fitter.fit(X.values.reshape(-1,1),y)
plt.plot(X,y,'o')
plt.plot(X,line_fitter.predict(X.values.reshape(-1,1)))
plt.title('Male linear regression')
plt.xlabel("Height (Inches)")
plt.ylabel("Weight (Pounds)")
plt.show()


#Male compute e
e=[]
for i in range(len(dfM)):
    wexpect=dfM['Height (Inches)'].iloc[i]
    e.append(dfM['Weight (Pounds)'].iloc[i]-line_fitter.predict([[wexpect]]))
dfMexp=pd.DataFrame(e)
dfMexp.columns=['e']


#Male compute Ze
ze=[]
for i in range(len(dfMexp)):
    ze.append((dfMexp['e'][i]-dfMexp['e'].mean())/dfMexp['e'].std())
dfMexp['Ze']=pd.DataFrame(ze)
plt.hist(dfMexp['Ze'].tolist(),bins=10,rwidth=0.9)
plt.title('Male Ze')
plt.xlabel('Ze')
plt.ylabel('frequency')
plt.show()

#Male alpha correction
alpha=1
correct=0
count=0
for i in range(len(dfM)):
    if dfMexp['Ze'][i]<-alpha:
        count+=1
        if dfM['BMI'].iloc[i]==0:
            correct+=1
            print("BMI 0-correct")
        else:
            print("BMI 0-not correct")
    elif dfMexp['Ze'][i]>alpha:
        count+=1
        if dfM['BMI'].iloc[i]==4:
            correct+=1
            print("BMI 4-correct")
        else:
            print("BMI 4-not correct")

print()
print('{}% correction' .format((correct/count)*100))

