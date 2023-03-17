import pickle

import clean_comment_util
from sklearn.neighbors import NearestCentroid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import VotingClassifier

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
Transforms the training data into a TF*IDF matrix
@param train_data: traing_data
@return tf_idf transformed matrix of data as well as the data transformer
'''


def transform_data(train_data):
    train_x_tfidf, transformer = tfidf_func(train_data, 1, 0.6, 'None')
    return train_x_tfidf, transformer


'''
Creates a TF*IDF matrix and returns the vectorizer for later use
@param data: training data
@param min_df: min_df to pass into TfidfVectorizer
@param max_df: max_df to pass into TfidfVectorizer
@param norm: a norm value to pass into TfidfVectorizer (if None, then it won't be passed)
@return returns tfidf matrix and vectorizer
'''


def tfidf_func(data, min_df, max_df, norm):
    if norm != 'None':
        vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, norm=norm)
    else:
        vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)
    txt_fitted = vectorizer.fit(data)
    txt_tranformed = txt_fitted.transform(data)
    X_tfidf = txt_tranformed.toarray()
    # terms = txt_fitted.get_feature_names_out()
    return X_tfidf, vectorizer


'''
A method to train all models and return an ensemble model
@param transformed_train_data: the training data which has already been transformed
@param train_labels: the corresponding training labels for the data
'''


def train_and_get_model(transformed_train_data, train_labels):
    # Best Params for NearestCentroid Model
    # token_value: TfidfVectorizer(max_df=0.6)
    # token_value__max_df: 0.6
    # token_value__min_df: 1

    # train_x_tfidf_nearest_centroid, transformer= tfidf_func(train_data, 1, 0.6, 'None')

    # Train nearest centroid model with Best Parameters
    clf1 = NearestCentroid(metric='cosine')
    # clf1.fit(train_x_tfidf_nearest_centroid, train_labels)

    # Best Params for NB Model
    # Pipeline = Pipeline(steps=[('vect', TfidfVectorizer()), ('clf', ComplementNB())])
    # Best parameters =
    # vect__max_df: 0.6
    # vect__min_df: 3
    # vect__norm: l1

    # create tfidf feature matrix for NB Model
    # train_x_tfidf_nb, transformer = tfidf_func(transformed_train_data, 3, 0.6, 'l1')
    clf2 = ComplementNB()
    # clf2.fit(train_x_tfidf_nb, train_labels)

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
    # create tfidf feature matrix for NearestCentroid Model

    # create tfidf feature matrix for Logistic Regression Model
    # train_x_tfidf_lr = tfidf_func(train_data, 1, 0.6, 'None')

    # Train Logistic Regression Model with Best Parameters
    normalizer = Normalizer(copy=False)
    svd = TruncatedSVD(n_components=109, random_state=516)
    # lg_train_data = normalizer.fit_transform(svd.fit_transform(transformed_train_data))
    clf3 = SGDClassifier(l1_ratio=0.9, loss='log', n_jobs=-1, penalty='elasticnet', random_state=961)
    # clf3.fit(lg_train_data, train_labels)

    # Lastly create an ensemble models with all 3 models
    eclf2 = VotingClassifier(estimators=[('nc', clf1), ('nb', clf2), ('sgd', clf3)], voting='hard')
    eclf2.fit(transformed_train_data, train_labels)

    return eclf2


'''
A method to classify a test comment
This method will clean and transform the data prior to predicting
@param test_data: the raw test data
@param data_transformer: the data transformer used to transform the train data
@param model: the model used to predict the test comment
@return the predicted class
'''


def classify(test_data, data_transformer, model):
    clean_test_data = clean_comment_util.clean_comment(test_data)
    test_data_array = [clean_test_data]
    test_x_tfidf = data_transformer.transform(test_data_array)

    predicted_class = model.predict(test_x_tfidf)
    print("The predicted class is:" + str(predicted_class))
    return toxic_app_classes[predicted_class[0]]


###############################Main Program Begins Here #########################

print("Welcome to the Toxic Comment Identifier App.")
train, train_labels = load_data()

transformed_data, data_transformer = transform_data(train)

model = train_and_get_model(transformed_data, train_labels)

test_data = input("Please enter a comment to classify or else 'q' to quit:").lower().strip()
while True:
    if test_data == 'q':
        print("Good Bye!")
        break
    else:
        print("This comment was classified as:" + str(classify(test_data, data_transformer, model)))
        test_data = input("Please enter a comment to classify or else 'q' to quit:")
