import numpy as np
import math


class Node:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.split_values = []

    def is_terminal(self):
        return len(self.children) == 0


class DecisionTree:
    def __init__(self):
        self.root = None

    def build(self, labels, data_points):
        self.root = self.build_tree_from_parent(None, labels, data_points, [], [])

    def predict(self, data_point):
        return self.predict_from_node(self.root, data_point)

    def print(self, label_names=None, feature_names=None, feature_values=None):
        self.print_tree_from_node(self.root, 0, label_names=label_names, feature_names=feature_names,
                                  feature_values=feature_values)

    def build_tree_from_parent(self, parent, labels, data_points, features, features_values):
        if parent is not None:
            pure_label, no_data_left = DecisionTree.pure_label_no_data_left(labels, data_points, features,
                                                                            features_values)
            if no_data_left:
                parent_name = features[-1]
                parent_value = features_values[-1]
                features.pop()
                features_values.pop()
                majority_label, majority_label_count = DecisionTree.return_majority(labels, data_points, features,
                                                                                    features_values)
                features.append(parent_name)
                features_values.append(parent_value)
                node = Node(majority_label)
                parent.children.append(node)
                return node
            if pure_label is not None:
                node = Node(pure_label)
                parent.children.append(node)
                return node
        left_features = DecisionTree.return_left_features(data_points, features)
        if len(left_features) == 1:
            left_feature = left_features[0]
            node = Node(left_feature)
            node.split_values = np.unique(data_points[:, left_feature])
            node = DecisionTree.make_terminal_nodes(node, labels, data_points, features, features_values)
            if parent is not None:
                parent.children.append(node)
            return node
        min_entropy = float("inf")
        min_entropy_index = -1
        for i in left_features:
            entropy = DecisionTree.con_entropy(labels, data_points, i, cols=features, cols_values=features_values)
            if entropy < min_entropy:
                min_entropy = entropy
                min_entropy_index = i
                if entropy == 0:
                    break
        node = Node(min_entropy_index)
        node.split_values = np.unique(data_points[:, min_entropy_index])
        if min_entropy == 0:
            node = DecisionTree.make_terminal_nodes(node, labels, data_points, features, features_values)
            if parent is not None:
                parent.children.append(node)
            return node
        features.append(min_entropy_index)
        for split_value in node.split_values:
            features_values.append(split_value)
            self.build_tree_from_parent(node, labels, data_points, features, features_values)
            features_values.pop()
        features.pop()
        if parent is not None:
            parent.children.append(node)
        return node

    def predict_from_node(self, root, data_point):
        if root.is_terminal():
            return root.name
        for i in range(len(root.children)):
            if root.split_values[i] == data_point[root.name]:
                return self.predict_from_node(root.children[i], data_point)

    def print_tree_from_node(self, node, cur_depth, label_names=None, feature_names=None, feature_values=None):
        if node.is_terminal():
            if label_names is not None:
                print(label_names[node.name])
            else:
                print(node.name)
            return
        print("\t" * cur_depth, end="**")
        if feature_names is not None:
            print(feature_names[node.name])
        else:
            print(node.name)
        for i in range(len(node.children)):
            print("\t" * cur_depth, end="\t*")
            end = "\n"
            if node.children[i].is_terminal():
                end = " --> "
            if feature_values is not None and feature_names is not None:
                print(feature_values[(feature_names[node.name], node.split_values[i])], end=end)
            else:
                print(node.split_values[i], end=end)
            self.print_tree_from_node(node.children[i], cur_depth + 2, label_names, feature_names, feature_values)

    @staticmethod
    def make_terminal_nodes(node, labels, data_points, features, features_values):
        features.append(node.name)
        for value in node.split_values:
            features_values.append(value)
            majority_label, majority_label_count = DecisionTree.return_majority(labels, data_points, features, features_values)
            if majority_label_count == 0:
                features_values.pop()
                features.pop()
                majority_label, majority_label_count = DecisionTree.return_majority(labels, data_points, features, features_values)
                features.append(node.name)
                features_values.append(value)
            terminal_node = Node(majority_label)
            node.children.append(terminal_node)
            features_values.pop()
        features.pop()
        return node

    @staticmethod
    def pure_label_no_data_left(labels, data_points, features, features_values):
        first_label = None
        for row in range(len(labels)):
            flag = False
            for j in range(len(features)):
                if data_points[row, features[j]] != features_values[j]:
                    flag = True
                    break
            if flag:
                continue
            if first_label is None:
                first_label = labels[row]
            elif first_label != labels[row]:
                return None, False
        if first_label is None:
            return None, True
        return first_label, False

    @staticmethod
    def con_entropy(labels, data_points, con_col, cols=[], cols_values=[]):
        result = 0
        label_values, label_counts = np.unique(labels, return_counts=True)
        con_col_values = np.unique(data_points[:, con_col])
        con_col_counts = {}
        for con_col_value in con_col_values:
            con_col_counts[con_col_value] = 0
        total_num = len(labels)
        joint_counts = {}
        for label_value in label_values:
            for col_value in con_col_values:
                joint_counts[(label_value, col_value)] = 0
        for row in range(total_num):
            flag = False
            for i in range(len(cols)):
                if data_points[row, cols[i]] != cols_values[i]:
                    flag = True
                    break
            if flag:
                continue
            con_col_counts[data_points[row, con_col]] += 1
            joint_counts[(labels[row], data_points[row, con_col])] += 1
        for con_col_value in con_col_values:
            temp = 0
            for label_value in label_values:
                if joint_counts[(label_value, con_col_value)] != 0:
                    prob = joint_counts[(label_value, con_col_value)] / con_col_counts[con_col_value]
                    temp -= prob * math.log(prob, 2)
            temp *= con_col_counts[con_col_value]
            result += temp
        result /= total_num
        return result

    @staticmethod
    def return_majority(labels, data_points, features, features_values):
        label_values = np.unique(labels)
        label_counts = {}
        for label_value in label_values:
            label_counts[label_value] = 0
        for row in range(len(labels)):
            flag = True
            for j in range(len(features)):
                if data_points[row, features[j]] != features_values[j]:
                    flag = False
                    break
            if not flag:
                continue
            label_counts[labels[row]] += 1

        if sum(label_counts.values()) == 0:
            pass
        majority_label = max(label_counts.keys(), key=(lambda key: label_counts[key]))
        majority_label_count = label_counts[majority_label]
        return majority_label, majority_label_count

    @staticmethod
    def return_left_features(data_points, selected_features):
        total_features_num = len(data_points[0])
        left_features_num = total_features_num - len(selected_features)
        left_features = []
        for j in range(total_features_num):
            if j not in selected_features:
                left_features.append(j)
                if len(left_features) == left_features_num:
                    return left_features
        return left_features


def read_data(file_name, label_col):
    result = np.genfromtxt(file_name, delimiter=',', dtype=str)
    return result[:, label_col], np.delete(result, label_col, 1)


def read_full_features_and_values_for_uci_mushroom_dataset(file_name):
    def fill_feature_values(value_key_pairs_str, feature_names, feature_values):
        value_key_pairs = value_key_pairs_str.split(",")
        for entry in value_key_pairs:
            if entry == '':
                continue
            [value, key] = entry.split('=')
            feature_values[(feature_names[-1], key)] = value

    file = open(file_name, "r")
    feature_names = []
    feature_values = {}
    while True:
        line = file.readline().strip()
        if line.startswith("7. Attribute Information"):
            break
    import re
    line = file.readline()
    while True:
        match_obj = re.match(r'^\s+\d+\. (.*)\:\s+(.+),?\n$', line)
        feature_names.append(match_obj.group(1))
        fill_feature_values(match_obj.group(2), feature_names, feature_values)
        flag = False
        while True:
            line = file.readline()
            if re.match(r'^\s+\d+\..*\n$', line):
                break
            match_obj = re.match(r'^\s+(.+),?\n?', line)
            if not match_obj:
                flag = True
                break
            fill_feature_values(match_obj.group(1), feature_names, feature_values)
        if flag:
            break
    file.close()
    return feature_names, feature_values


def predict_all(decision_tree, data_points):
    predictions = []
    for i in range(data_points.shape[0]):
        predictions.append(decision_tree.predict(data_points[i, :]))
    return predictions


def accuracy(true_labels, predicted_labels):
    correct_num = 0
    for i in range(len(true_labels)):
        if true_labels[i] == predicted_labels[i]:
            correct_num += 1
    return correct_num / len(true_labels) * 100


if __name__ == "__main__":
    training_labels, training_data_points = read_data('mush_train.data', 0)
    test_labels, test_data_points = read_data("mush_test.data", 0)
    feature_names, feature_values = read_full_features_and_values_for_uci_mushroom_dataset("agaricus-lepiota.names")
    label_names = {"p": "poisonous", "e": "edible"}
    decision_tree = DecisionTree()
    decision_tree.build(training_labels, training_data_points)
    decision_tree.print(label_names=label_names, feature_names=feature_names, feature_values=feature_values)
    # decision_tree.print(label_names=label_names, feature_names=feature_names)
    # print(list(zip(test_labels, predict_all(decision_tree, test_data_points))))
    print("Test accuracy: {}%".format(accuracy(test_labels, predict_all(decision_tree, test_data_points))))
