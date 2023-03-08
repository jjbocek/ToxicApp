import numpy as np

def get_number_of_classes(classes):
    return len(np.unique(classes))

"""A Training Function for the Rocchio Classification Algorithm
:param data: an np array of the training data formatted as a doc-term frequency matrix,
a ttd*idf matrix or else a doc2vec vector matrix
:param labels: an np array of the corresponding labels for the data
:returns: the prototype vectors for each class (a.k.a. label)
"""


def rocchio_train(train, labels):
    # the prototype will be a dictionary with unique class labels as keys and on dimensional arrays representing
    # the prototype.
    prototype = {}
    #tf_idf_train = compute_tf_idf_matrix(train)
    num_labels = get_number_of_classes(labels)
    #print("The number of classes:" + str(num_labels))

    # add each class to the prototype dictionary
    # the class is the key
    # a one dimensional array equal to the length of a term vector array is initialized with all 0s
    for label in labels:
        term_vector_array = np.zeros(np.shape(len(train[0])), dtype=float)
        prototype[label] = term_vector_array

    length = len(train)
    for i in range(length):
        # first figure out which class this entry belongs to
        label = labels[i]

        # next sum up the entire document term vector
        prototype[label] = prototype.get(label) + np.array(train[i])

    return prototype

"""A Classifier Function for the Rocchio Classification Algorithm
:param data: a dictionary of prototype vectors where the class is a key and the value is the vector
for the corresponding class.
:param instance: the test instance to classify
:returns: the predicted class
"""


def rocchio_classifier(prototype, test_instance):
    m = -2
    predicted_class = -1  # intialize the predicted class to -1 by default
    for classLabel in prototype:
        # (compute similarity to prototype vector)
        # use cosine similarity
        # cosine distance is 1 - (dot product / L2 norm)
        term_vector = prototype[classLabel]
        prototype_norm = np.array([np.linalg.norm(term_vector) for i in range(len(term_vector))])
        test_instance_norm = np.linalg.norm(test_instance)

        sims = np.dot(term_vector, test_instance) / (prototype_norm * test_instance_norm)

        # get the maximum cosSim
        max_cos_sim = np.max(sims)

        if max_cos_sim > m:
            m = max_cos_sim
            predicted_class = classLabel

    return predicted_class