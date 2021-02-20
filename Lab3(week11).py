
# Linear Regression
import pandas as pd
import numpy as np
import warnings
from matplotlib import pyplot as plt
import sklearn.model_selection as skms
import sklearn.linear_model as sklm

warnings.filterwarnings(action='ignore')

#store a dataset into data
data=pd.read_csv('linear_regression_data.csv')

dataSet=np.array(data)
#Distance
x=dataSet[:,0].reshape(-1,1)
#Delivery Time
y=dataSet[:,1].reshape(-1,1)


#train and prediction
x_train,x_test,y_train,y_test = skms.train_test_split(x,y,
test_size = 0.2, random_state=200)


#linear regression
reg = sklm.LinearRegression()
#fit the data
reg.fit(x_train,y_train)
#predict the data
y_predict = reg.predict(x_test)


#for evaluate
#draw the diagram through scatter on the plot in differ from test and train
plt.scatter(x_test,y_predict, color="RED")
plt.scatter(x_test,y_test,color="BLUE")
plt.scatter(x_train,y_train,color="BLACK")

px = np.array([x_test.min()-1,x_test.max()+1])
#from the x , predict the y
py=reg.predict(px[:,np.newaxis])
#draw linear regression plot
plt.plot(px,py,color="GREEN")
plt.title("Linear Regression")
plt.show()




########################################################################################################
# Decision Tree
import pandas as pd
import warnings

warnings.filterwarnings(action='ignore')

data = pd.read_csv('decision_tree_data.csv')


# calculate entropy of dataset
# target_column is specifcation of the target column
def entropy(target_column):
    elements, counts = np.unique(target_column, return_counts=True)
    entropy = np.sum(
        [(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return entropy


# calculate information gain of dataset
# data is dataset
# split_attribute_name is the name of feature that calculate information gain
# target_name is the name of target feature
def InfoGain(data, split_attribute_name, target_name="interview"):
    # Calculate the entropy of the total dataset
    total_entropy = entropy(data[target_name])

    # Calculate the values and the corresponding counts for the split attribute
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)

    # Calculate the weighted entropy
    Weighted_Entropy = np.sum(
        [(counts[i] / np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
         for i in range(len(vals))])

    # Calculate the information gain
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain


# use ID3 algorithm to make tree
# data is the dataset for which the ID3 algorithm should be run
# original_data is the original dataset needed to calculate the mode target feature value of the original dataset
# features is the feature space of the dataset, it is needed for the recursive call since during the tree growing process
# target_attribute_name is the name of target attribute
# parent_node_class is the value or class of the mode target feature value of the parent node for a specific node,
# it is needed for the recursive call
def ID3(data, original_data, features, target_attribute_name="interview", parent_node_class=None):
    # If all target_values have the same value, return this value
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    # If the dataset is empty, return the mode target feature value in the original dataset
    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[
            np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]

    # If the feature space is empty, return the mode target feature value of the direct parent node
    # the direct parent node is that node which has called the current run of the ID3 algorithm and hence
    # the mode target feature value is stored in the parent_node_class variable

    elif len(features) == 0:
        return parent_node_class

    # If none of the above holds true, grow the tree

    else:
        # Set the default value for this node --> The mode target feature value of the current node
        parent_node_class = np.unique(data[target_attribute_name])[
            np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]

        # Select the feature which best splits the dataset
        item_values = [InfoGain(data, feature, target_attribute_name) for feature in
                       features]  # Return the information gain values for the features in the dataset
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        # Create the tree structure. The root gets the name of the feature (best_feature) with the maximum information
        # gain in the first run
        tree = {best_feature: {}}

        # Remove the feature with the best inforamtion gain from the feature space
        features = [i for i in features if i != best_feature]

        # Grow a branch under the root node for each possible value of the root node feature

        for value in np.unique(data[best_feature]):
            value = value

            # Split the dataset along the value of the feature with the largest information gain and therwith create sub_datasets
            sub_data = data.where(data[best_feature] == value).dropna()

            # Call the ID3 algorithm for each of those sub_datasets with the new parameters
            # Here the recursion comes in
            subtree = ID3(sub_data, data, features, target_attribute_name, parent_node_class)

            # Add the sub tree, grown from the sub_dataset to the tree under the root node
            tree[best_feature][value] = subtree

        return (tree)


# dataset divided into a training and a testing set
# the testing data as well as the tree model
def train_test_split(data):
    training_data = data.iloc[:80].reset_index(drop=True)
    # We drop the index respectively relabel the index
    # starting form 0, because we do not want to run into errors regarding the row labels / indexes
    testing_data = data.iloc[80:].reset_index(drop=True)
    return training_data, testing_data


training_data = train_test_split(data)[0]
testing_data = train_test_split(data)[1]

tree = ID3(training_data, training_data, training_data.columns[:-1])
print(tree)




##########################################################################################################
# KNN
import pandas as pd
import numpy as np
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

warnings.filterwarnings(action='ignore')

# store a data set into the data
data = pd.read_csv('knn_data.csv')

# longitude
data['longitude'] = pd.to_numeric(data['longitude'])
# latitude
data['latitude'] = pd.to_numeric(data['latitude'])
# store data without target, lang
x = data.drop(columns=['lang'])
x = x.values
# lang
y = data['lang']

# train and predict
kf = KFold(n_splits=5)

print("divided by 5, k=5: ")

for train_index, test_index in kf.split(x):
    # store the data
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # x_train, x_test, y_train, y_test, y_train, y_test
    x_train = pd.DataFrame({'longitude': x_train[:, 0], 'latitude': x_train[:, 1]})
    x_test = pd.DataFrame({'longitude': x_test[:, 0], 'latitude': x_test[:, 1]})
    y_train = pd.DataFrame({'original': y_train})
    y_test = pd.DataFrame({'original': y_test})
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # k-nearest neighbors
    knn = KNeighborsClassifier(n_neighbors=5)
    # fit
    knn.fit(x_train, y_train['original'])
    # predict value for x_test
    predict = knn.predict(x_test)

    # save the predicted values due to the data, you have learned about each attribute and place them in the results
    result = pd.DataFrame(
        {'longitude': x_test['longitude'], 'latitude': x_test['latitude'], 'original': y_test['original'],
         'predict': predict})

# print the result and accuarancy
print(result)
print("** The accuracy of prediction:", knn.score(x_test, y_test))
print()
print()
print()

# train and predict
kf = KFold(n_splits=2)

print("divided by 2, k=2: ")

for train_index, test_index in kf.split(x):
    # store the data
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # x_train, x_test, y_train, y_test, y_train, y_test
    x_train = pd.DataFrame({'longitude': x_train[:, 0], 'latitude': x_train[:, 1]})
    x_test = pd.DataFrame({'longitude': x_test[:, 0], 'latitude': x_test[:, 1]})
    y_train = pd.DataFrame({'original': y_train})
    y_test = pd.DataFrame({'original': y_test})
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # k-nearest neighbors
    knn = KNeighborsClassifier(n_neighbors=2)
    # fit
    knn.fit(x_train, y_train['original'])
    # predict value for x_test
    predict = knn.predict(x_test)

    # save the predicted values due to the data, you have learned about each attribute and place them in the results
    result = pd.DataFrame(
        {'longitude': x_test['longitude'], 'latitude': x_test['latitude'], 'original': y_test['original'],
         'predict': predict})

# print the result and accuarancy
print(result)
print("** The accuracy of prediction:", knn.score(x_test, y_test))
print()
print()
print()

# train and predict
kf = KFold(n_splits=10)

print("divided by 10, k=10: ")

for train_index, test_index in kf.split(x):
    # store the data
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # x_train, x_test, y_train, y_test, y_train, y_test
    x_train = pd.DataFrame({'longitude': x_train[:, 0], 'latitude': x_train[:, 1]})
    x_test = pd.DataFrame({'longitude': x_test[:, 0], 'latitude': x_test[:, 1]})
    y_train = pd.DataFrame({'original': y_train})
    y_test = pd.DataFrame({'original': y_test})
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # k-nearest neighbors
    knn = KNeighborsClassifier(n_neighbors=10)
    # fit
    knn.fit(x_train, y_train['original'])
    # predict value for x_test
    predict = knn.predict(x_test)

    # save the predicted values due to the data, you have learned about each attribute and place them in the results
    result = pd.DataFrame(
        {'longitude': x_test['longitude'], 'latitude': x_test['latitude'], 'original': y_test['original'],
         'predict': predict})

# print the result and accuarancy
print(result)
print("** The accuracy of prediction:", knn.score(x_test, y_test))


