# Decison Tree Classifier by M.Mobin.Teymourpour & E.Samiei (4022129, 4012236)

import numpy as np
import pandas as pd


# read the data from the csv files:
read_feature_train = pd.read_csv("feature_train.csv").values.tolist()
read_labels_train = pd.read_csv("label_train.csv").values.tolist()
read_feature_test = pd.read_csv("feature_test.csv").values.tolist()
read_labels_test = pd.read_csv("label_test.csv").values.tolist()
features_row = pd.read_csv("feature_train.csv", header=None).values.tolist()[0]
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
    # Tree constructor
    def __init__(self, max_depth = None, features_row = None):
        self.depth = 0
        self.max_depth = max_depth
        self.root = None
        self.features_row = features_row
    
    def get_depth(self):
        return self.depth
    
    def create_tree(self, data, labels, node = None):
        #if node is None, create a root node
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

        if max(label_counts) / len(labels) < 0.9 and node.level < self.max_depth: 

            best_split = self.get_best_split(data, labels) # it's a dictionary taht gets the best feature and it's children
            node.children = best_split["children"] # set children based of best split
            node.info_gain = best_split["max_ig"] # set info gain of the node
            node.feature = best_split["feature"] # set feature of the node

            # create a child node for each unique feature value recursively:
            for child in node.children:
                child.level = node.level + 1
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
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.tree = Tree(3 * len(read_feature_train[0]) // 4, features_row)
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

        print(len(results))

        return results
    
    # calculate the accuracy of the model
    def accuracy(self, labels, labels_predicted):
        true_predict = 0
        for i in range(len(labels)):
            if labels[i] == labels_predicted[i]:
                true_predict += 1

        return (true_predict / len(labels)) * 100
    
def plot_tree(tree):

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
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=2000, node_color=color_map, font_size=7)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.show()

classifier = DTreeClassifier(read_feature_train, read_labels_train)        
labels_predicted = classifier.predict_all(read_feature_test, 3 * len(features_row) // 4)
accuracy = classifier.accuracy(read_labels_test, labels_predicted)
print(accuracy)