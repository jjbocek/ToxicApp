from gensim.models import doc2vec
from nltk.tokenize import word_tokenize
from numpy import savetxt

"""
A method to get a list of doc2vec vectors based on the model and dataframe passed in
The dataframe should just consist of 1 column which contains the document or comment
to be vectorized

:param df : the dataframe which contains the document or comment to be vectorized
:return a list of vectors corresponding to the data passed in
"""
def get_doc2vec_vectors(model, data):
    # #using default values for now
    tokenized_comments = tokenize_comments(data)
    tagged_documents = get_tagged_documents(tokenized_comments)

    # build the vocabulary
    # input a list of documents
    model.build_vocab(x for x in tagged_documents)

    # Train the model
    model.train(tagged_documents, total_examples = model.corpus_count, epochs = model.epochs)

    #print("Inferring "+str(len(tokenized_comments)) +" comments into doc2vec vectors.")
    vectors = infer_vectors(model, tokenized_comments)
    return vectors

"""
A method that infers a list of vectors from a trained Doc2Vec model
: param model : a Doc2Vec model which is already trained with vocab built
: param input : a data frame to infer Doc2Vec vectors from
: param save_file_name [OPTIONAL] : If a string is provided,
the vectors will be saved using this file name.
"""
def infer_vectors(model, tokenized_comments):
    #print("Inferring "+str(len(tokenized_comments)) +" comments into doc2vec vectors.")
    vectors = []
    for comment in tokenized_comments:
        #count = count + 1
        #print("Vectorizing: "+str(count)+" comment.")
        vectors.append(model.infer_vector(comment))

    #print("Created "+str(len(vectors)) + " doc2vec vectors.")
    #save to file if a file name is present
    #if save_file_name != "":
    #    print("Saving vectors to file: " + str(save_file_name))
    #    savetxt(save_file_name, vectors)

    return vectors

"""
A function to tokenize all data in a dataframe
:param data: a dataframe containing comments to tokenize
"""

def tokenize_comments(dataframe):
    data = []
    for row in dataframe:
        data.append(tokenize_each_comment(row))
    return data
"""
A function to tokenize a single comment
:param data: a single comment to tokenize
"""
def tokenize_each_comment(comment):
    temp = []
    for j in word_tokenize(comment):
        temp.append(j)
    return temp

"""
A function to generate a list of tagged documents to train a
Doc2Vec model
:param list_of_tokenized_comments: A list of tokenized comments
"""
def tagged_document(list_of_tokenized_comments):
  for x, ListOfWords in enumerate(list_of_tokenized_comments):
    yield doc2vec.TaggedDocument(ListOfWords, [x])

"""
A function to get tagged documents from
a list of tokenized comments
"""
def get_tagged_documents(list_of_tokenized_comments):
    return list(tagged_document(list_of_tokenized_comments))
