import pandas as pd
import nltk
from sklearn import metrics
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    corpus = []
    for i in range(len(text)):
        r = re.sub('[^a-zA-Z]', ' ', text[i])
        r = r.lower()
        r = r.split()
        r = [word for word in r if word not in stopwords.words('english')]
        r = ' '.join(r) 
        corpus.append(r)
    return corpus

def main():
    path = r'./labeledtext.csv'
    datasur = pd.read_csv(path)  # <- Uzupełnij ścieżkę do pliku CSV
    print(datasur.head(5))
    # Stworzenie dataframe z próbką danych 1000 sztuk z każdego labelu
    dataright = datasur[datasur['label'] == 'right'].head(1000)
    dataleft = datasur[datasur['label'] == 'left'].head(1000)
    data = pd.concat([dataright, dataleft])
    data.columns = ['title', 'text', 'label']
    data = data.dropna()

    text = list(data['text'])

    # Preprocessing
    corpus = preprocess_text(text)
    data["text"] = corpus

    print(data.head())

    # Stworzenie dwóch setów
    X = data['text']
    y = data['label']

    # Rozdział datasetu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)

    print('Training Data:', X_train.shape)
    print('Testing Data:', X_test.shape)

    # Bag of Words
    cv = CountVectorizer()
    X_train_cv = cv.fit_transform(X_train)
    print('Shape of training data after CountVectorizer:', X_train_cv.shape)

    # Trenowanie
    lr = LogisticRegression()
    lr.fit(X_train_cv, y_train)

    X_test_cv = cv.transform(X_test)

    # Predykcje
    predictions = lr.predict(X_test_cv)

    # Stworzenie macierzy z wynikami
    df = pd.DataFrame(metrics.confusion_matrix(y_test, predictions), index=['left', 'right'], columns=['left', 'right'])
    print(df)

    # Obliczyć accuracy
    score = accuracy_score(y_test, predictions)
    print('Accuracy:', score)

if __name__ == "__main__":
    main()
