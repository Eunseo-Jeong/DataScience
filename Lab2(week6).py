import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

#Read CSV File
df=pd.read_csv('bmi_data_lab3.csv')

extremelyweak=0
weak=0
normal=0
overweight=0
obesity=0
df0_h=[]
df0_w=[]
df1_h=[]
df1_w=[]
df2_h=[]
df2_w=[]
df3_h=[]
df3_w=[]
df4_h=[]
df4_w=[]

#
for i in range(len(df)):
    if df['BMI'][i]==0:
        extremelyweak+=1
        df0_h.append(df['Height (Inches)'][i])
        df0_w.append(df['Weight (Pounds)'][i])
    elif df['BMI'][i]==1:
        weak+=1
        df1_h.append(df['Height (Inches)'][i])
        df1_w.append(df['Weight (Pounds)'][i])
    elif df['BMI'][i]==2:
        normal+=1
        df2_h.append(df['Height (Inches)'][i])
        df2_w.append(df['Weight (Pounds)'][i])
    elif df['BMI'][i]==3:
        overweight+=1
        df3_h.append(df['Height (Inches)'][i])
        df3_w.append(df['Weight (Pounds)'][i])
    elif df['BMI'][i]==4:
        obesity+=1
        df4_h.append(df['Height (Inches)'][i])
        df4_w.append(df['Weight (Pounds)'][i])
print('extremelyweak={}, weak={}, normal={}, overweight={}, obesity={}'
     .format(extremelyweak,weak,normal,overweight,obesity))


#Histogram BMI=0
plt.hist(df0_h,bins=10,rwidth=0.9)
plt.title('Histogram of normal(BMI=0)')
plt.xlabel('Height')
plt.ylabel('Number of Student')
plt.show()
plt.hist(df0_w,bins=10,rwidth=0.9)
plt.title('Histogram of normal(BMI=0)')
plt.xlabel('Weight')
plt.ylabel('Number of Student')
plt.show()


#Histogram BMI=1
plt.hist(df1_h,bins=10,rwidth=0.9)
plt.title('Histogram of normal(BMI=1)')
plt.xlabel('Height')
plt.ylabel('Number of Student')
plt.show()
plt.hist(df1_w,bins=10,rwidth=0.9)
plt.title('Histogram of normal(BMI=1)')
plt.xlabel('Weight')
plt.ylabel('Number of Student')
plt.show()


#Histogram BMI=2
plt.hist(df2_h,bins=10,rwidth=0.9)
plt.title('Histogram of normal(BMI=2)')
plt.xlabel('Height')
plt.ylabel('Number of Student')
plt.show()
plt.hist(df2_w,bins=10,rwidth=0.9)
plt.title('Histogram of normal(BMI=2)')
plt.xlabel('Weight')
plt.ylabel('Number of Student')
plt.show()


#Histogram BMI=3
plt.hist(df3_h,bins=10,rwidth=0.9)
plt.title('Histogram of normal(BMI=3)')
plt.xlabel('Height')
plt.ylabel('Number of Student')
plt.show()
plt.hist(df3_w,bins=10,rwidth=0.9)
plt.title('Histogram of normal(BMI=3)')
plt.xlabel('Weight')
plt.ylabel('Number of Student')
plt.show()

#Histogram BMI=4
plt.hist(df4_h,bins=10,rwidth=0.9)
plt.title('Histogram of normal(BMI=4)')
plt.xlabel('Height')
plt.ylabel('Number of Student')
plt.show()
plt.hist(df4_w,bins=10,rwidth=0.9)
plt.title('Histogram of normal(BMI=4)')
plt.xlabel('Weight')
plt.ylabel('Number of Student')
plt.show()


#Standard Scaling
df2=df[df['BMI']==2]
df_2=df2[['Height (Inches)', 'Weight (Pounds)']]
scaler=preprocessing.StandardScaler()
scaled_df=scaler.fit_transform(df_2)
scaled_df=pd.DataFrame(scaled_df,columns=['Height','Weight'])
fig, (ax1,ax2)=plt.subplots(ncols=2,figsize=(6,5))
ax1.set_title('Before Scaling')
sns.kdeplot(df_2['Height (Inches)'],ax=ax1)
sns.kdeplot(df_2['Weight (Pounds)'],ax=ax1)
ax2.set_title('After Standard Scaler')
sns.kdeplot(scaled_df['Height'],ax=ax2)
sns.kdeplot(scaled_df['Weight'],ax=ax2)
plt.show()


#Minmax Scaling
df2=df[df['BMI']==2]
df_2=df2[['Height (Inches)', 'Weight (Pounds)']]
scaler=preprocessing.MinMaxScaler()
scaled_df=scaler.fit_transform(df_2)
scaled_df=pd.DataFrame(scaled_df,columns=['Height','Weight'])
fig, (ax1,ax2)=plt.subplots(ncols=2,figsize=(6,5))
ax1.set_title('Before Scaling')
sns.kdeplot(df_2['Height (Inches)'],ax=ax1)
sns.kdeplot(df_2['Weight (Pounds)'],ax=ax1)
ax2.set_title('After MinMax Scaler')
sns.kdeplot(scaled_df['Height'],ax=ax2)
sns.kdeplot(scaled_df['Weight'],ax=ax2)
plt.show()


#Robust Scaling
df2=df[df['BMI']==2]
df_2=df2[['Height (Inches)', 'Weight (Pounds)']]
scaler=preprocessing.RobustScaler()
scaled_df=scaler.fit_transform(df_2)
scaled_df=pd.DataFrame(scaled_df,columns=['Height','Weight'])
fig, (ax1,ax2)=plt.subplots(ncols=2,figsize=(6,5))
ax1.set_title('Before Scaling')
sns.kdeplot(df_2['Height (Inches)'],ax=ax1)
sns.kdeplot(df_2['Weight (Pounds)'],ax=ax1)
ax2.set_title('After Robust Scaler')
sns.kdeplot(scaled_df['Height'],ax=ax2)
sns.kdeplot(scaled_df['Weight'],ax=ax2)
plt.show()



## of NAN for each column
print("Sex NAN: ",len(df.loc[df["Sex"] == "NAN"]))
print("Age NAN: ",len(df.loc[df["Age"] == "NAN"]))
print("Height NAN: ",len(df.loc[df["Height (Inches)"] == "NAN"]))
print("Weight NAN: ",len(df.loc[df["Weight (Pounds)"] == "NAN"]))
print("BMI NAN: ",len(df.loc[df["BMI"] == "NAN"]))
