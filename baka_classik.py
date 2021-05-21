import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer as Vectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, log_loss, \
    cohen_kappa_score, balanced_accuracy_score, make_scorer

class NLPTopicClassifier(object):
    """MLdata-PeopleData-bakaset.csv are answers from questionnaires. Each row represents all text answers from one user.
    First column is company id, followed by two text columns for cs_text - sometimes empty, and en-text
    then follows individual category. There's a 0/1 depending whether the category isn't/is mentioned in the text."""

    def __init__(self, train_file="data/MLdata-PeopleData.csv", train_pct=0.8, labels=["temperature", "air",
        "acoustics", "light", "facilities", "ergonomics", "culture", "coffee_snacks", "focus", "cleanliness", "design",
        "relax", "meetings"], stopwords_extend= ["sometimes", "already", "also", "could", "either", "especially",
        "etc", "even", "every", "example", "still", "us", "usually", "would", "whether", "10", "100"], train_df=None):
        if train_df is None:
            self.df = pd.read_csv(train_file)
        else:
            self.df = train_df

        self.classifiers = np.array(labels)  # list of used classifiers, aka topics.
        self.nlp = {}  # dict for each classifier has a dict: "train", "test" (data), "classifier" (model), ...
        for cls in self.classifiers:
            self.nlp[cls] = {}
            self.nlp[cls]["text"] = {}

        # this format depends on cols in our CSV data file
        for label, i in zip(labels, range(1, len(labels)+1)):
            train, test = train_test_split(self.df, train_size=train_pct, random_state=0, shuffle=True, stratify=self.df[label])
            self.nlp[label]["text"]["train"] = train.iloc[:, 0].values
            self.nlp[label]["text"]["test"] = test.iloc[:, 0].values
            self.nlp[label]["train"] = train.iloc[:, i].values  #train and test labels for nlp[cls]["text"]["train/test"]
            self.nlp[label]["test"] = test.iloc[:, i].values

        #prepare stopwords
        self.stops = [self.preprocess_cleanup(word) for word in stopwords.words('english')]
        self.stops.extend(stopwords_extend)

    def train(self):
        """
        Transforms train text into matrix of TF-IDF features and trains a classifier for each topic in self.classifiers.
        After train, self.nlp[topic]["classifier"] contains a trained classifier on that topic.
        """
        self.vectorizers = {}
        for topic in self.nlp.keys():
            self.vectorizers[topic] = Vectorizer(
                                    max_features=9500,
                                    min_df=10, #accuracy depends a lot on this, which words get to be features
                                    analyzer='word',
                                    stop_words=self.stops, #TODO improve stopwords - use other dataset? see from trained features if looks correct!
                                    vocabulary=None, #TODO manually improve, hint vocabulary? Make sure it's just hints, and not hardcoded
                                    #TODO use stemming from NLTK https://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes
                                    decode_error='replace',
                                    strip_accents='unicode',
                                    ngram_range=(1,2), #we use 1-3-words as feautures, ie "air conditioning"
                                    preprocessor=self.preprocess_cleanup)

        self.Xs = {}

        for topic in self.nlp.keys():
            self.Xs[topic] = self.vectorizers[topic].fit_transform(self.nlp[topic]["text"]["train"]).toarray()
            self.nlp[topic]["classifier"] = RandomForestClassifier(n_estimators=200, random_state=0)
            self.nlp[topic]["classifier"].fit(self.Xs[topic], self.nlp[topic]["train"]) # train separate classifiers for each topic on the whole text (X) and specific labels

    def misclassified_matrix(self):
        """
        Show misclassifications and confusion matrix.

        @return nothing, prints the misclassifications
        """
        for topic in self.classifiers:
            data = self.vectorizers[topic].transform(self.nlp[topic]["text"]["test"]).toarray()
            predictions = self.nlp[topic]["classifier"].predict(data)
            print("'{}' confusion matrix: \n{}".format(topic, confusion_matrix(self.nlp[topic]["test"], predictions)))

            misclassified = np.where(self.nlp[topic]["test"] != predictions)

            for num in misclassified[0]:
                print("text: {} \npredicted: {} really: {}\n".format(self.nlp[topic]["text"]["test"][num],
                                                                     predictions[num], self.nlp[topic]["test"][num]))


    def eval(self):
        """
        For each classifier prints a cross validated score of chosen metrics.
        (if interested in performance, use train_pct=0.99, cross_val_score() will split the data)
        """

        for topic in self.classifiers:
            kappa_scorer = make_scorer(cohen_kappa_score)
            log_loss_scorer = make_scorer(log_loss)
            scorings = ['accuracy', 'balanced_accuracy', log_loss_scorer, 'f1_macro', 'f1_weighted', kappa_scorer]
            for sco in scorings:
                results = cross_val_score(estimator=self.nlp[topic]["classifier"], X=self.Xs[topic],
                                          y=self.nlp[topic]["train"], cv=5, scoring=sco)
                print("{} results from cross val with {}".format(topic, sco))
                print("{:.3f}".format(np.mean(results)))

            """Prints non crossed valued scores"""
            # data = self.vectorizer.transform(self.nlp[topic]["text"]["test"]).toarray()
            # predictions = self.nlp[topic]["classifier"].predict(data)
            # print("'{}' accuracy: {:.3f} (train size: {} / test {})".format(topic, float(accuracy_score(self.nlp[topic]["test"], predictions)),
            # np.sum(self.nlp[topic]["train"]), np.sum(self.nlp[topic]["test"]) )) #compare expected and given predictions on test set
            # print("'{}' balanced accuracy: {:.3f}".format(topic, float(balanced_accuracy_score(self.nlp[topic]["test"], predictions))))
            # print("'{}' log_loss: {:.3f}".format(topic, float(log_loss(self.nlp[topic]["test"], predictions))))
            # print("'{}' macro f1: {:.3f}".format(topic, float(f1_score(self.nlp[topic]["test"], predictions, average='macro'))))
            # print("'{}' weighted f1: {:.3f}".format(topic, float(f1_score(self.nlp[topic]["test"], predictions, average='weighted'))))
            # print("'{}' kappa: {:.3f}".format(topic, float(cohen_kappa_score(self.nlp[topic]["test"], predictions))))


    def label(self, sentence):
        """
        Label a sentence using trained classifiers.

        @arg sentence - text to be classified (str)

        @return binary array if the 'sentence' contains the 'topic' from self.classifiers.
        """
        assert isinstance(sentence, str), "Input must be a single sentence, string."

        labels = []
        for topic in self.classifiers:
            data = self.vectorizers[topic].transform([sentence]).toarray()
            preds = self.nlp[topic]["classifier"].predict(data)
            #print("{} preds {}".format(topic, preds))
            labels.append(int(preds[0])) #somehow, one label is always being '1' (str), instead of 1 (int)

        return np.array(labels)


    def preprocess_cleanup(self, text):
        """
        Preprocesses the text for nlp.

        @param text - text to be processed (str)

        @return string without special characters, only single spaces, lower-cased, etc..
        """
        # Remove all the special characters
        processed_sentence = re.sub(r'\W+', ' ', str(text))
        # remove all single characters
        #processed_sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_sentence)
        # Remove single characters from the start
        #processed_sentence = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_sentence)
        # Substituting multiple spaces with single space
        processed_sentence = re.sub(r'\s+', ' ', processed_sentence, flags=re.I)
        # Removing prefixed 'b'
        processed_sentence = re.sub(r'^b\s+', '', processed_sentence)
        # Converting to Lowercase
        processed_sentence = processed_sentence.lower()

        return processed_sentence

if __name__ == "__main__":
    # NLP classification:
    #nlp = GrnstTopicNLPClassifier(train_file="data/MLdata-PeopleData-bakaset.csv", train_pct=0.7)
    nlp = NLPTopicClassifier()
    nlp.train()
    nlp.misclassified_matrix()
    #nlp.eval()
    # txt = "I'm always cold and hate the air conditioning."
    # print("Labels for text '{}' are {} = {}. ".format(t, nlp.classifiers, nlp.label(t)))

