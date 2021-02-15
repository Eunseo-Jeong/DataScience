import pandas as pd
import numpy as np

# Decision Tree
# dataset
data = pd.DataFrame({
    "District": ["Suburban", "Suburban", "Rural", "Urban", "Suburban", "Suburban", "Suburban", "Rural", "Rural",
                 "Rural", "Urban", "Urban", "Urban", "Urban"],
    "House Type": ["Detached", "Semi-detached", "Semi-detached", "Detached", "Detached", "Semi-detached",
                   "Semi-detached", "Detached", "Detached", "Detached", "Detached", "Detached", "Detached", "Detached"],
    "Income": ["High", "High", "Low", "Low", "Low", "Low", "Low", "Low", "Low", "Low", "Low", "Low", "Low", "Low"],
    "Previous Customer": ["No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "No", "No", "No"],
    "Outcome": ["Nothing", "Respond", "Respond", "Nothing", "Nothing", "Respond", "Respond", "Respond", "Respond",
                "Respond", "Nothing", "Respond", "Respond", "Respond"]})
features = data[["District", "House Type", "Income", "Previous Customer", "Outcome"]]
target = data["Outcome"]
print(data, '\n')


# calculate entropy of dataset
# target_column is specification of the target column
def entropy(target_column):
    elements, counts = np.unique(target_column, return_counts=True)
    entropy = np.sum(
        [(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return entropy


# calculate information gain of dataset
# data is dataset
# split_attribute_name is the name of feature that calculate information gain
# target_name is the name of target feature
def InfoGain(data, split_attribute_name, target_name="Outcome"):
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
def ID3(data, original_data, features, target_attribute_name="Outcome", parent_node_class=None):
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



# KNN Algorithm
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings(action='ignore')

# dataset
data = pd.DataFrame({
    "Height(cm)": [158, 158, 158, 160, 160, 163, 163, 160, 163, 165, 165, 165, 168, 168, 168, 170, 170, 170],
    "Weight(kg)": [58, 59, 63, 59, 60, 60, 61, 64, 64, 61, 62, 65, 62, 63, 66, 63, 64, 68],
    "T_shirt_size": ["M", "M", "M", "M", "M", "M", "M", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L"]})
features = data[["Height(cm)", "Weight(kg)", "T_shirt_size"]]
print(data, '\n')

# Load and save data
df_s = np.array(data.T_shirt_size)
# The result depends on height and weight
df_xy = data[["Height(cm)", "Weight(kg)"]]
data_xy = df_xy.as_matrix()

# Categorized
height = int(input('Enter your height:'))
weight = int(input('Enter your weight:'))
target = [height, weight]


# Create classification targets and categories
def data_set():
    size = len(df_xy)
    class_target = np.tile(target, (size, 1))
    class_z = np.array(df_s)
    return df_xy, class_target, class_z


# data_set() function recall to create dataset
dataset, class_target, class_z = data_set()


# Euclidean distance calculation
def classify(dataset, class_target, class_category, k):
    # a difference of two points
    diffMat = class_target - dataset

    # square for difference
    sqDiffMat = diffMat ** 2

    # sum of squares for difference
    row_sum = sqDiffMat.sum(axis=1)

    # Final Distance = Consensus square root for the square of the car
    distance = np.sqrt(row_sum)

    # Sorting in ascending order of distance
    sortDist = distance.argsort()

    # neighbor k selection
    class_result = {}
    for i in range(k):
        c = class_category[sortDist[i]]
        class_result[c] = class_result.get(c, 0) + 1

    return class_result


# function call
k = int(input('Enter the value of k :'))
# classify() function call print(class_result)
class_result = classify(data_xy, class_target, class_z, k)


# Output function as a result of classification
def resultprint(class_result):
    weightt = heightt = 0

    for c in class_result.keys():
        if c == 'M':
            weightt = class_result[c]
        elif c == 'L':
            heightt = class_result[c]

    if weightt > heightt:
        result = "Size: M"
    elif heightt > weightt:
        result = "Size L"
    else:
        result = "Change the value of k."

    return result


print(resultprint(class_result))

# visualization
import matplotlib.pyplot as plt

he = data["Height(cm)"]
we = data["Weight(kg)"]
t = data["T_shirt_size"]

for i in range(len(he)):
    if t[i] == 'M':
        plt.scatter(he[i], we[i], color='b')
    else:
        plt.scatter(he[i], we[i], color='g')
plt.scatter(height, weight, marker='*', color='r')
plt.show()