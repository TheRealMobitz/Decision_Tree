import numpy as np
import pandas as pd
import math
from sklearn import tree
import graphviz
import random
from collections import Counter
read_feature_train = pd.read_csv("feature_train.csv").values.tolist()
read_labels_train = pd.read_csv("label_train.csv").values.tolist()
read_feature_test = pd.read_csv("feature_test.csv").values.tolist()
read_labels_test = pd.read_csv("label_test.csv").values.tolist()
features_row = pd.read_csv("feature_train.csv", header = None).values.tolist()[0]
read_labels_train = [item for sublist in read_labels_train for item in sublist]
read_labels_test = [item for sublist in read_labels_test for item in sublist]


class Node: 
    def __init__(self, feature = None, level = None, children = None, info_gain = None, data = None, labels = None, based_on = None, value = None):
        # for decision node
        self.feature = feature
        self.level = level
        self.children = children if children else []
        self.info_gain = info_gain
        self.data = data if data is not None else []
        self.labels = labels if labels is not None else []
        self.based_on = based_on
        # for leaf node
        self.value = value
class Tree:
    def __init__(self, max_depth = None, features_row = None):
        self.depth = 0
        self.max_depth = max_depth
        self.root = None
        self.features_row = features_row
    
    def get_depth(self):
        return self.depth
    
    def create_tree(self, data, labels, node = None):
        if node == None:
            self.root = Node(level = 1, data = data, labels = labels)
            self.depth += 1
            node = self.root
        label_counts = self.get_label_details(labels)["label_counts"]
        dominant_class =  self.get_label_details(labels)["majority_label"]   
        # Create a leaf node if any stopping condition is met
        if max(label_counts) / len(labels) < 0.95 and node.level < self.max_depth:
            best_split = self.get_best_split(data, labels)
            node.children = best_split["children"]
            node.info_gain = best_split["max_ig"]
            node.feature = best_split["feature"]
            for child in node.children:
                child.level = node.level + 1
                if child.level > self.depth:
                    self.depth += 1
                self.create_tree(child.data, child.labels, child)
        else:
           node.value =  dominant_class
    def get_label_details(self, labels):
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        majority_label = unique_labels[np.argmax(label_counts)]
        dict_labels = {"unique_labels" : unique_labels, "label_counts" : label_counts, "majority_label" : majority_label}
        return dict_labels
    
    def get_best_split(self, data, labels):
        max_ig = -float("inf")
        best_feature = None 
        best_children = None
        for feature in range(len(self.features_row)):
            children = []
            unique_features = np.unique(np.array(data)[:, feature])
            for unifeat in unique_features:
                node = Node()
                children.append(node)
                for row in data:
                    if row[feature] == unifeat:
                        node.data.append(row)
                        node.labels.append(labels[data.index(row)])
                        node.based_on = unifeat
            current_ig = self.info_gain_calc(self.entropy_calc(labels), children, labels)
            if current_ig > max_ig:
                max_ig = current_ig
                best_feature = feature
                best_children = children
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
class DTreeClassifire:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.tree = Tree(3 * len(features_row) // 4, features_row)
        self.tree.create_tree(data, label)
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
    def accuracy(self, labels, labels_predicted):
        true_predict = 0
        for i in range(len(labels)):
            if labels[i] == labels_predicted[i]:
                true_predict += 1
        return (true_predict/len(labels)) * 100

class RForestClassifire:
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


classifire = DTreeClassifire(read_feature_train, read_labels_train) 
labels_predicted = classifire.predict_all(read_feature_test, len(features_row) // 2)
accuracy = classifire.accuracy(read_labels_test, labels_predicted)
print("accuracy : "+ accuracy)

RForest = RForestClassifire(read_feature_train, read_labels_train, 6, 3)
estimators = RForest.build_forest()
accuracy = RForest.accuracy(read_labels_test, estimators)
print("forest accuracy: " + accuracy)




                     
        


