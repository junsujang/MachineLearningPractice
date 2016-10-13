from math import log
import copy

def make_tree(dataset, attributes) :
    """
        Create a recursive tree to build the hypothesis

    :param dataset: dataset to learn from (could be reduced dataset)
    :param labels: list which matches the title of the attributes
    :return: dictionary representation of the decision tree
    """
    label_list = [data[-1] for data in dataset]
    # stop if the dataset has all the same label
    if label_list.count(label_list[0]) == len(dataset) :
        return label_list[0]

    # return the labels with majority when there is no more attribute
    # to be looked up (only label left in the dataset)
    if len(dataset[0] == 1) :
        return majority_label(dataset)

    # Make a root node for recursion
    optimal_attr_index = choose_optimal_attribute(dataset)
    optimal_attr_title = attributes[optimal_attr_index]
    tree = {optimal_attr_title:{}}
    del(attributes[optimal_attr_index])

    # For each unique value under chosen attribute, create sub-tree
    attr_list = [data[optimal_attr_index] for data in dataset]
    attr_unique_vals = set(attr_list)
    for val in attr_unique_vals :
        sub_labels = copy.deepcopy(attributes)
        reduced_dataset = reduce_dataset(dataset, optimal_attr_index, val)
        tree[optimal_attr_title][val] = make_tree(reduced_dataset, sub_labels)

    return tree

def majority_label(dataset) :
    #TODO implement this function
    return

def choose_optimal_attribute(dataset) :
    """
    for each attribute, calculate the information gain, which is
    the difference between the original entropy and the entropy
    when the dataset has been splitted by the attribute values

    :param dataset: dataset for the program to learn from
    :return: the most optimal attribute that would reduce the entropy
    """
    num_attr = len(dataset[0] - 1) # subtract 1 for label
    num_data = len(dataset)
    base_entropy = calculate_entropy(dataset)
    most_info_gain = 0
    optimal_attr_index = -1
    for i in range(num_attr) :
        attr_list = [data[i] for data in dataset]
        attr_unique_vals = set(attr_list)
        new_entropy = 0
        # for each attribute unique values see how much entropy is caused
        # and add them to finalize the total entropy when splitted by
        # this specific attribute
        for val in attr_unique_vals :
            sub_dataset = reduce_dataset(dataset, i, val)
            prob = len(sub_dataset) / num_data
            new_entropy += prob * calculate_entropy(sub_dataset)
        # compare and choose the attr with the most info gain
        info_gain = base_entropy - new_entropy
        if info_gain > most_info_gain :
            most_info_gain = info_gain
            optimal_attr_index = i
    return optimal_attr_index


def reduce_dataset(dataset, axis, value) :
    """
    
    :param dataset: all the dataset given to be reduced
    :param axis: the index of the attribute being checked
    :param value: the searched value
    :return: reduced dataset with vectors whose attribute located at axis is
                that of the value
    """
    reduced_dataset= []
    for vector in dataset :
        if vector[axis] == value :
            # if a value of an attribute matches, add it to the reduced dataset
            reduced_vector = vector[:axis]
            reduced_vector.extend(vector[axis+1:])
            reduced_dataset.append(reduced_vector)
    return reduced_dataset

def calculate_entropy(dataset) :
    """
    Calculates the Shannon Entropy of given dataSet.
        The Shannon entropy equation provides a way to estimate the average
        minimum number of bits needed to encode a label
        based on the frequency of the labels.

    Follows the pseudo code :
        H = -(sum([p(label) * log(p(label)) for label in dataset])

    :param dataset :
        A list of list of data [attr1, attr2, ... , label]

    """

    num_entries = len(dataset)
    label_counts = {}
    for vector in dataset :
        # the label is at the last index of the data set
        current_label = vector[-1]
        if current_label not in label_counts :
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    # Calculate the entropy
    entropy = 0.0
    for label in label_counts :
        # Calculate probability of each label within the dataset
        prob_of_label = label_counts[label]/num_entries
        # Since the entropy is the negative of the sum of all probability,
        # simply subtract it
        entropy -= prob_of_label * log(prob_of_label, 2)
    return entropy


