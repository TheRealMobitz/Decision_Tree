# Decison Tree Classifier by M.Mobin.Teymourpour & E.Samiei (4022129, 4012236)

# for reading the data from the csv files
import numpy as np
import pandas as pd

# for drawing the decision tree
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx
import matplotlib.pyplot as plt

# for random forest generator
import math
import random
from collections import Counter

# read the data from the csv files:
read_feature_train = pd.read_csv("feature_train.csv").values.tolist()
read_labels_train = pd.read_csv("label_train.csv").values.tolist()
read_feature_test = pd.read_csv("feature_test.csv").values.tolist()
read_labels_test = pd.read_csv("label_test.csv").values.tolist()
features_row = pd.read_csv("feature_train.csv", header=None).values.tolist()[0]
read_labels_train = [item for sublist in read_labels_train for item in sublist]
read_labels_test = [item for sublist in read_labels_test for item in sublist]


class Node: 
    def __init__(self, feature = None, level = None, children = None, info_gain = None, data = None, labels = None, based_on = None, value = None,  parent = None):
        # for decision node
        self.feature = feature
        self.level = level
        self.children = children if children else []
        self.info_gain = info_gain
        self.data = data if data is not None else []
        self.labels = labels if labels is not None else []
        self.based_on = based_on
        self.parent = parent
        # for leaf node
        self.value = value
class Tree:
    # Tree constructor
    def __init__(self, max_depth = None, features_row = None):
        self.depth = 0
        self.max_depth = max_depth
        self.root = None
        self.features_row = features_row
        self.threshold = threshold_setter(read_labels_train)
    
    def get_depth(self):
        return self.depth
    
    def create_tree(self, data, labels, node = None):
        # if node is None, create a root node
        if node == None:
            self.root = Node(level = 1, data = data, labels = labels)
            self.depth += 1
            node = self.root
        
        label_counts = self.get_label_details(labels)["label_counts"] # number unique labels for each label
        dominant_class =  self.get_label_details(labels)["majority_label"] # majority label in each node 

        '''
        Create children nodes if conditions are met
        gets maximum count of labels and devides it by total number of labels
        and if the level of the current node is less than the max_depth
        then create a decision node:
        '''

        if max(label_counts) / len(labels) < self.threshold and node.level < self.max_depth: 

            best_split = self.get_best_split(data, labels) # it's a dictionary taht gets the best feature and it's children
            node.children = best_split["children"] # set children based of best split
            node.info_gain = best_split["max_ig"] # set info gain of the node
            node.feature = best_split["feature"] # set feature of the node

            # create a child node for each unique feature value recursively:
            for child in node.children:
                child.level = node.level + 1
                child.parent = node
                if child.level > self.depth:
                    self.depth += 1
                self.create_tree(child.data, child.labels, child)

        # create a leaf node if stopping condition is met
        else:
           node.value =  dominant_class #if a node had an value it means it's a leaf node in other words, value is the result

    # get the unique labels, majority label and the count of each label
    def get_label_details(self, labels):
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        majority_label = unique_labels[np.argmax(label_counts)]
        # create a dictionary for returning
        dict_labels = {"unique_labels" : unique_labels, "label_counts" : label_counts, "majority_label" : majority_label}

        return dict_labels
    
    # get the best split for the current node
    def get_best_split(self, data, labels):
        max_ig = -float("inf") # set the max info gain to negative infinity bc in initializing we want is as minimum as possible
        best_feature = None 
        best_children = None

        for feature in range(len(self.features_row)):
            children = []
            unique_features = np.unique(np.array(data)[:, feature]) # get the unique features for the current feature

            # for each unique feature create a child node and append it to the children list
            for unifeat in unique_features:
                node = Node(based_on = unifeat)
                children.append(node)
                for row in data:
                    if row[feature] == unifeat:
                        node.data.append(row)
                        node.labels.append(labels[data.index(row)]) 

            # get the info gain for the current feature and it's children
            current_ig = self.info_gain_calc(self.entropy_calc(labels), children, labels)

            # if the current info gain is greater than the max info gain, set the current info gain as the max info gain
            if current_ig > max_ig:
                max_ig = current_ig
                best_feature = feature
                best_children = children

        # return the best feature, it's children and the max info gain
        return {"feature": best_feature, "children": best_children, "max_ig": max_ig}    
    
    def entropy_calc(self, labels):
        entropy = 0
        total = len(labels)
        label_counts = self.get_label_details(labels)["label_counts"]
        for label_count in label_counts:
            probiblity =  label_count / total
            entropy += -probiblity * np.log2(probiblity)
        return entropy
    
    def info_gain_calc(self, parent_entropy, children, labels):
        sigma = 0
        for child in children:
            sigma += (len(child.labels) / len(labels)) * self.entropy_calc(child.labels)
        return parent_entropy - sigma
    
    # Traverse the tree using DFS
    def traverse(self):
        def dfs(node):
            if node is not None:
                yield node
                for child in node.children:
                    yield from dfs(child)

        return dfs(self.root) 


class DTreeClassifier:
    # constructor:
    def __init__(self, data, label, max_depth = None):
        self.data = data
        self.label = label
        self.max_depth = max_depth
        self.tree = Tree(max_depth ,features_row)
        self.tree.create_tree(data, label)

    # predict the label of a single person
    def predict(self, data, depth):
        node = self.tree.root

        while node.level < depth and node.value is None:
            feature_index = data[node.feature]
            found = False # if a child is found
            for child in node.children:
                if child.based_on == feature_index:
                    node = child
                    found = True
                    break

            if not found:  # No matching child found
                return self.tree.get_label_details(node.labels)["majority_label"]
            
        if node.value is not None:
            return node.value
        
        else:
            return self.tree.get_label_details(node.labels)["majority_label"]
    
    # predict the label of all people in the data
    def predict_all(self, data, depth):
        results = []
        for person in data:
            results.append(self.predict(person, depth))

        return results
    
    # calculate the accuracy of the model
    def accuracy(self, labels, labels_predicted):
        true_predict = 0
        for i in range(len(labels)):
            if labels[i] == labels_predicted[i]:
                true_predict += 1

        return (true_predict / len(labels)) * 100

class RForestClassifier:
    def __init__(self, data, label, number_of_estimator, max_depth):
        self.data = data
        self.label = label
        self.number_of_estimator = number_of_estimator
        max_depth = max_depth
        self.tree = None
        self.estimators = []
        self.build_forest()
    def build_forest(self):
        for _ in range(self.number_of_estimator):
            random_columns = random.sample(range(0, len(features_row) - 1), int(math.sqrt(len(features_row))))
            temp_data = []  
            for _ in range(len(self.data)):
                temp_data.append(self.data[random.randint(0, len(self.data) - 1)])
            temp_data = [[row[i] for i in random_columns] for row in temp_data]
            self.tree = Tree(3 * len(features_row) // 4, random_columns)
            self.tree.create_tree(temp_data, self.label)
            labels_predicted = self.predict_all(temp_data, len(features_row) // 2)
            self.estimators.append(labels_predicted)
        return self.estimators

    def predict(self, data, depth):
        node = self.tree.root
        while node.level < depth and node.value is None:
            feature_index = data[node.feature]
            found = False
            for child in node.children:
                if child.based_on == feature_index:
                    node = child
                    found = True
                    break
            if not found:
                return self.tree.get_label_details(node.labels)["majority_label"]
        if node.value is not None:
            return node.value
        else:
            return self.tree.get_label_details(node.labels)["majority_label"]
    def predict_all(self, data, depth):
        results = []
        for person in data:
            results.append(self.predict(person, depth))
        return results
    def accuracy(self, labels, estimators):
        transposed_estimators = list(map(list, zip(*estimators)))
        labels_predicted = []
        for column in transposed_estimators:
            counts = Counter(column)
            most_common_number = max(counts, key=counts.get)
            labels_predicted.append(most_common_number)
        true_predict = 0
        for i in range(len(labels)):
            if labels[i] == labels_predicted[i]:
                true_predict += 1
        return (true_predict/len(labels)) * 100




def plot_tree(tree, accuracy = None):

    G = nx.DiGraph() # Create a directed graph
    labels = {} # Create a dictionary to store the labels of the nodes
    edge_labels = {} # Create a dictionary to store the labels of the edges
    color_map = [] # Create a list to store the colors of the nodes to differentiate between internal and leaf nodes

    # Traverse the tree using DFS and add the nodes and edges to the graph 
    for node in tree.traverse():
        unique_labels, label_counts = np.unique(node.labels, return_counts=True) # Get the unique labels and their counts
        label_counts_dict = dict(zip(unique_labels, label_counts))
        
        # Add the decision node to the graph:
        if node.feature is not None:
            labels[node] = f'name: {features_row[node.feature]}\ninfo_gain: {node.info_gain}\n'
            
            # Add the label counts to the node label and then set the color
            for i in range(len(unique_labels)):
                labels[node] += f'label {unique_labels[i]} : {label_counts_dict.get(i, i)}\n'
            color_map.append('skyblue')

            # Add the edge to the graph:
            if node.parent is not None:
                edge_labels[(node.parent, node)] = node.based_on

        # Add the leaf node to the graph:
        elif node.value is not None:
            labels[node] = f'value: {node.value}\ninformation gain: {node.info_gain}\n'
            
            for i in range(len(unique_labels)):
                labels[node] += f'label {unique_labels[i]} : {label_counts_dict.get(i, i)}\n'

            color_map.append('lightgreen')

        # Add the nodes and edges to the graph
        if node.parent is not None:
            G.add_edge(node.parent, node)

    # Draw the graph:
    pos = graphviz_layout(G, prog='dot')
    plt.figure(figsize=(10, 10), facecolor = 'black')
    plt.title(f'Decision Tree\n Accuracy: {accuracy:.2f}%')
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=2000, node_color=color_map, font_size=7)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.show()

def threshold_setter(labels_list):
    # Get the unique labels and their counts
    _, label_counts = np.unique(labels_list, return_counts=True)

    # Set the threshold based on the maximum label count
    if max(label_counts) / len(labels_list) < 0.8:
        print(0.8)
        return 0.8
    elif max(label_counts) / len(labels_list) < 0.85:
        print(0.85)
        return 0.85
    elif max(label_counts) / len(labels_list) < 0.9:
        print(0.9)
        return 0.9
    elif max(label_counts) / len(labels_list) < 0.95:
        print(0.95)
        return 0.95
    else:
        print(0.98)
        return 0.98

def depth_list_creator(features_row):
    # Create a list of depth values to be used for the decision tree
    depth_list = []
    num = len(features_row)

    # Add the odd numbers from 1 to 3/4 of the number of features
    for i in range(4, (3 * num // 4) + 1, 2):
        depth_list.append(i)

    if (3 * num // 4) + 1 not in depth_list:
        depth_list.append((3 * num // 4) + 1)

    return depth_list


def calculate_best_max_depth(data, labels, max_depth_values, validation_ratio=0.2):

    # Initialize the best depth and accuracy
    best_depth = None
    best_accuracy = -1
    
    # Split the data into training and validation sets
    validation_size = int(len(data) * validation_ratio)
    training_data = data[validation_size:]
    training_labels = labels[validation_size:]
    validation_data = data[:validation_size]
    validation_labels = labels[:validation_size]
    
    # Calculate the accuracy for each depth value
    for depth in max_depth_values:
        
        dtree = DTreeClassifier(training_data, training_labels, depth)

        predictions = dtree.predict_all(validation_data, depth)
        accuracy = dtree.accuracy(validation_labels, predictions)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_depth = depth
    
    return best_depth

classifier = DTreeClassifier(read_feature_train, read_labels_train, calculate_best_max_depth(read_feature_train, read_labels_train, depth_list_creator(features_row))) # create a decision tree classifier    
labels_predicted = classifier.predict_all(read_feature_test, len(features_row) // 2) # predict the labels of the test data
accuracy = classifier.accuracy(read_labels_test, labels_predicted) # calculate the accuracy of the model
plot_tree(classifier.tree, accuracy) # draws the tree =))

# RForest = RForestClassifier(read_feature_train, read_labels_train, 6, calculate_best_max_depth(read_feature_train, read_labels_train, depth_list_creator(features_row)))
# estimators = RForest.build_forest()
# accuracy = RForest.accuracy(read_labels_test, estimators)
# print("forest accuracy: ", accuracy)