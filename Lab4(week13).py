###################################################
###################################################
#################        import       ######################
###################################################
###################################################

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import warnings

warnings.filterwarnings(action='ignore')

###################################################
###################################################
################        Read File       #####################
###################################################
###################################################


iris = pd.read_csv('iris\\Iris_test_dataset.csv', encoding='utf-8')  # read file and set data
labels = iris['Species']  # label with species

# one-hot encoding
# set Iris-setosa to 0
# set Iris-versicolor to 1
# set Iris-virginica to 2
for i in range(len(labels)):
    if (labels[i] == 'Iris-setosa'):
        labels[i] = 0
    elif (labels[i] == 'Iris-versicolor'):
        labels[i] = 1
    elif (labels[i] == 'Iris-virginica'):
        labels[i] = 2

###################################################
###################################################
################        Data Test       #####################
###################################################
###################################################

# set X_test // drop feature about Species and Id and look each rows
# we see only SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
X_test = iris.drop(["Species", "Id"], axis=1)

# set Y_pred which predict from X_test data set
Y_pred = []

print("Starting Test\n")

for i in range(10):
    # Loading bagging data sequential
    String = "iris\Iris_train_datasets\Iris_bagging_dataset (" + str(i + 1) + ").csv"
    sample = pd.read_csv(String, encoding='utf-8')

    #     print(sample)

    X_train = sample.drop(["Species", "Id"], axis=1)
    Y_train = sample["Species"]

    model = DecisionTreeClassifier(criterion='entropy')  # Using DecisionTreeClassifier method in sklearn

    model.fit(X_train, Y_train)  # Model tranining

    Y_pred.append(model.predict(X_test))  # Predict of testing dataset and append to Y_pred

print()
print("Finish")

# ###################################################
# ###################################################
# ###############        Find Accurancy       ##################
# ###################################################
# ###################################################

correct_pred = []

for i in range(len(iris)):
    list_pred = [0, 0, 0]
    for j in range(10):
        if (Y_pred[j][i] == 'Iris-setosa'):
            list_pred[0] = list_pred[0] + 1
        if (Y_pred[j][i] == 'Iris-versicolor'):
            list_pred[1] = list_pred[1] + 1
        if (Y_pred[j][i] == 'Iris-virginica'):
            list_pred[2] = list_pred[2] + 1
    correct_pred.append(list_pred.index(max(list_pred)))

# Compare the original value

count = 0
for i in range(len(iris)):
    if (labels[i] == correct_pred[i]):
        count = count + 1

###################################################
###################################################
##############        Print Accurancy       ##################
###################################################
###################################################

print("\nAccuracy: ", (count / 150) * 100, "%")

###################################################
###################################################
##################        Finish       #####################
###################################################
###################################################
