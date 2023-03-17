import pickle

import clean_comment_util
from sklearn.neighbors import NearestCentroid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline

toxic_app_classes = {1: 'non-toxic', 2: 'toxic', 3: 'severe toxic'}

'''
Loads a clean version of the training data from file
'''


def load_data():
    file_object = open('../Final_Project/clean_data1.p', 'rb')
    clean_data = pickle.load(file_object)
    train_ds = clean_data[0]

    # pre-process the corpus
    train_comments_df = train_ds['comment_text']
    train_labels_df = train_ds['toxicity_level']

    return train_comments_df, train_labels_df


'''
A method to train all models and return an ensemble model
@param transformed_train_data: the training data which has already been transformed
@param train_labels: the corresponding training labels for the data
'''


def train_and_get_model(train_data, train_lab):
    # Best Params for NearestCentroid Model
    # token_value: TfidfVectorizer(max_df=0.6)
    # token_value__max_df: 0.6
    # token_value__min_df: 1

    # Nearest centroid model with Best Parameters
    clf1 = make_pipeline(TfidfVectorizer(min_df=1, max_df=0.6), NearestCentroid(metric='cosine'))

    # Best Params for NB Model
    # Pipeline = Pipeline(steps=[('vect', TfidfVectorizer()), ('clf', ComplementNB())])
    # Best parameters =
    # vect__max_df: 0.6
    # vect__min_df: 3
    # vect__norm: l1

    # NB model with best parameters
    clf2 = make_pipeline(TfidfVectorizer(min_df=3, max_df=0.6, norm='l1'), ComplementNB())

    # Best Params for Logistic Regression
    # clf__l1_ratio: 0.9
    # clf__loss: log
    # clf__n_jobs: -1
    # clf__penalty: elasticnet
    # reduce_dim: Pipeline(steps=[('truncatedsvd', TruncatedSVD(n_components=109)),
    #                 ('normalizer', Normalizer(copy=False))])
    # token_value: TfidfVectorizer(max_df=0.6)
    # token_value__max_df: 0.6
    # token_value__min_df: 1

    # Train Logistic Regression Model with Best Parameters
    clf3 = make_pipeline(TfidfVectorizer(min_df=1, max_df=0.6),
                         TruncatedSVD(n_components=112, random_state=516),
                         Normalizer(copy=False),
                         SGDClassifier(l1_ratio=0.9, loss='log', n_jobs=-1, penalty='elasticnet', random_state=961))

    # Lastly create an ensemble models with all 3 models
    eclf = VotingClassifier(estimators=[('nc', clf1), ('nb', clf2), ('sgd', clf3)], voting='hard')
    eclf.fit(train_data, train_lab)

    return eclf


'''
A method to classify a test comment
This method will clean and transform the data prior to predicting
@param test_data: the raw test data
@param data_transformer: the data transformer used to transform the train data
@param model: the model used to predict the test comment
@return the predicted class
'''


def classify(test_data, model):
    clean_test_data = clean_comment_util.clean_comment(test_data)
    clean_test_data_array = [clean_test_data]
    predicted_class = model.predict(clean_test_data_array)
    # print("The predicted class is:" + str(predicted_class))
    return toxic_app_classes[predicted_class[0]]


'''
Main Program Beings Here
'''

print("Welcome to the Toxic Comment Identifier App.")
train, train_labels = load_data()

ensemble_model = train_and_get_model(train, train_labels)

raw_input = input("Please enter a comment to classify or else 'q' to quit:").lower().strip()
while True:
    if raw_input == 'q':
        print("Good Bye!")
        break
    else:
        print("This comment was classified as: " + str(classify(raw_input, ensemble_model)))
        raw_input = input("Please enter a comment to classify or else 'q' to quit:")
