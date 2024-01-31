import numpy as np
class Node:
    def __init__(self, feature_index=None, threshold=None, children = None, info_gain=None, value=None):
        
        
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.children = children
        self.info_gain = info_gain
        # for leaf node
        self.value = value
class Tree:
    def __init__(self, min_samples_split, max_depth):
        self.depth = 0
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
    