import numpy as np
import pandas as pd

read_feature_train = pd.read_csv("feature_train.csv")
read_labels_train = pd.read_csv("label_train.csv")


class Node: 
    def __init__(self, feature = None, level = None, children = None, info_gain = None, data = None, labels = None, value = None):
        # for decision node
        self.feature = feature
        self.level = level
        self.children = children
        self.info_gain = info_gain
        self.data = data
        self.labels = labels
        # for leaf node
        self.value = value
class Tree:
    def __init__(self):
        self.depth = 0
        self.max_depth = None
        
    
    def get_depth(self):
        return self.depth
    
    def create_tree(self, data, labels, node = None):
        if node == None:
            self.root = Node(level = 1, data = data, labels = labels)
            self.depth += 1
            node = self.root
        label_counts = self.get_label_details(labels)["label_counts"]
        unique_labels = self.get_label_details(labels)["unique_labels"]    
        dominant_class = unique_labels[np.argmax(label_counts)]
        # Create a leaf node if any stopping condition is met
        if (max(label_counts) / len(labels)) < 0.95 and node.level < self.max_depth:
            best_split = self.get_best_split(data, labels)
            node.children = best_split["children"]
            node.info_gain = best_split["max_ig"]
            node.feature = best_split["feature"]
            for child in node.children:
                child.level = node.level + 1
                if child.level > self.depth:
                    self.depth += 1
                self.create_tree(child)
        else:
           node.value =  dominant_class
    def get_label_details(self, labels):
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        dict_labels = {"unique_labels" : unique_labels, "label_counts" : label_counts}
        return dict_labels
    
    def get_best_split(self, data, labels):
        
        max_ig = -float("inf")
        best_feature = None 
        best_children = None
        for feature in range(data.shape[1]):
            children = []
            unique_features = np.unique(data[:, feature])
            for unifeat in unique_features:
                node = Node()
                children.append(node)
                for row in data:
                    if row[feature] == unifeat:
                        node.data.append(row)
                        node.labels.append(labels[data.index(row)])  
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

tree = Tree()
tree.create_tree()        
