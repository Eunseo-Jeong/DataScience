###################################
###################################
###########      import     ##############
###################################
###################################
import pandas as pd
import numpy as np
import warnings
import math
import random
from matplotlib import pyplot as plt
import sklearn.model_selection as skms
import sklearn.linear_model as sklm
warnings.filterwarnings(action='ignore')

###################################
###################################
###########      get data    #############
###################################
###################################

Path='C:\\Users\\JuHwan\\Desktop\\데이터과학\\hw4\\4-1\\linear_regression_data.csv'

data=pd.read_csv(Path)
print(data)

###################################
###################################
####      Save each attribute separately     ####
###################################
###################################

Data_np = np.array(data) #set the data into Data_np
X=Data_np[:,0].reshape(-1,1) #set the X with distance
Y=Data_np[:,1].reshape(-1,1) #set the Y with delivery time


###################################
###################################
########     Train & Predict     ##########
###################################
###################################
A = math.ceil(abs(random.randint(2,10))) #set random number between 2~10

#Data preprocessing
X_train,X_test,Y_train,Y_test = skms.train_test_split(X,Y,
test_size = 0.2, random_state=A)

#Regression
reg = sklm.LinearRegression() #using Linear Regression
reg.fit(X_train,Y_train) #fit the data

Y_predict = reg.predict(X_test) #predict the data

#Evaluation
#And draw the diagram
#set the point
plt.scatter(X_test , Y_predict, color="RED")
plt.scatter(X_test,Y_test,color="BLUE")
plt.scatter(X_train,Y_train,color='black')
px = np.array([X_test.min()-1,X_test.max()+1])
#from the x , predict the y
py=reg.predict(px[:,np.newaxis])
plt.plot(px,py,color='black') #set the color
plt.xlabel("")
plt.title("Linear Regression")
plt.ylabel("")
plt.show()
