
import pickle
import rocchio_model
import doc2vec_methods
from gensim.models import doc2vec
import clean_comment_util
import numpy as np
from sklearn.neighbors import NearestCentroid

toxic_app_classes = {1: 'non-toxic', 2: 'toxic', 3: 'severe toxic'}
d2v_model = doc2vec.Doc2Vec(dm=0, vector_size=500, window=2, hs=1, negative=0)
rocchio_prototypes = {}
nearest_centroid = NearestCentroid()

def load_data():
    file_object = open('../clean_data1.p', 'rb')
    clean_data = pickle.load(file_object)
    train_ds = clean_data[0]

    # pre-process the corpus
    train_comments_df = train_ds['comment_text']
    train_labels_df = train_ds['toxicity_level']

    return train_comments_df, train_labels_df


def get_feature_matrix(feature_extraction, data):
    print("Selected feature extraction: "+str(feature_extraction))
    if feature_extraction == 'td_idf':
        #generate a td_idf matrix of the training data
        return 0
    elif feature_extraction == 'doc2vec':
        train_vectors = doc2vec_methods.get_doc2vec_vectors(d2v_model, data)
        return train_vectors
    else:
        print("Unrecognized feature extraction method.  Using td_idf.")
        return 0

def train_models(model, comments, labels):
    print("Training a "+model+" model.")
    if model == "gd":
        # train gd model
        return 0
    elif model == "kmeans":
        # train kmeans model
        return 0
    elif model == "rocchio":
        global rocchio_prototypes
        rocchio_prototypes = rocchio_model.rocchio_train(comments, labels)
    elif model == "nb":
        # train nb model
        return 0
    elif model == "knn":
        # train knn model
        return 0
    else:
        print("Unknown model inputted, using gradiant descent instead.")
        return 0

def transform_test_data(test_data, feature_extraction):
    if feature_extraction == 'td_idf':
        # generate a td_idf vector of this comment
        return 0
    elif feature_extraction == 'doc2vec':
        # infer a doc2vec vector of this comment
        return d2v_model.infer_vector(doc2vec_methods.tokenize_each_comment(test_data))


#generate a doc2vec matrix of the training data
def classify(test_data, feature_extraction, model):
    clean_test_data = clean_comment_util.clean_comment(test_data)
    test_vector = transform_test_data(clean_test_data, feature_extraction)

    if model == "gd":
        # train gd model
        return 0
    elif model == "kmeans":
        # train kmeans model
        return 0
    elif model == "rocchio":
        global rocchio_prototypes
        predicted_class = rocchio_model.rocchio_classifier(rocchio_prototypes, test_vector)
        print("The predicted class was: "+str(predicted_class))
    elif model == "nb":
        # train nb model
        return 0
    elif model == "knn":
        # train knn model
        return 0
    return toxic_app_classes[predicted_class]


################################Main Program Begins Here #########################
print("Welcome to the Toxic Comment Identifier App.")
train, train_labels = load_data()

feature_transformation_method = input("Please select a feature extraction method to use.  Type 'td_idf' or 'doc2vec'.")
training_matrix = get_feature_matrix(feature_transformation_method, train)

model_selection = input("Please select a model to use. Your choices are:\n"
                        "Gradient Descent/Logistic Regression (type 'gd')\n"
                        "Kmeans (type 'kmeans')\n"
                        "Rocchio Classification(type 'rocchio')\n"
                        "Naive Bayes(type 'nb')\n"
                        "KNN (type 'knn')")

train_models(model_selection, training_matrix, np.array(train_labels))
test_data = input("Please enter a comment to classify or else 'q' to quit:")
while True:
    if test_data == 'q':
        print("Good Bye!")
        break
    else:
        print("This comment was classified as:" + str(classify(test_data, feature_transformation_method, model_selection)))
        test_data = input("Please enter a comment to classify or else 'q' to quit:")

