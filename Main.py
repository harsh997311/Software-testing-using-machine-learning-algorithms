from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.svm import SVC
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation

main = tkinter.Tk()
main.title("An Experimental Study for Software Quality Prediction with Machine Learning Methods")
main.geometry("1200x1200")

global filename
global X, Y
global X_train, X_test, y_train, y_test
global dataset
accuracy = []
precision = []
recall = []
fscore = []

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]]
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) // nGraphPerRow
    plt.figure(num=None, figsize=(6 * nGraphPerRow, 8 * nGraphRow), dpi=80, facecolor='w', edgecolor='k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if not np.issubdtype(type(columnDf.iloc[0]), np.number):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation=90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.show()

def uploadDataset():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n\n")
    dataset = pd.read_csv(filename)
    text.insert(END, str(dataset.head))
    plotPerColumnDistribution(dataset, 40, 5)

def preprocess():
    text.delete('1.0', END)
    global X, Y
    fig, ax = plt.subplots()
    sns.lineplot(data=dataset.isnull().sum())
    fig.autofmt_xdate()
    dataset.fillna(0, inplace=True)
    cols = ['QualifiedName', 'Name', 'Complexity', 'Coupling', 'Size', 'Lack of Cohesion']
    le = LabelEncoder()
    for col in cols:
        dataset[col] = pd.Series(le.fit_transform(dataset[col].astype(str)))
    Y = dataset.values[:, 2]
    dataset.drop(['Complexity'], axis=1, inplace=True)
    X = dataset.values
    X = normalize(X)
    text.insert(END, str(X) + "\n")
    plt.show()

def featureSelection():
    global X, Y
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test
    text.insert(END, "Total features found in dataset before applying feature selection algorithm = " + str(X.shape[1]) + "\n")
    pca = PCA(n_components=30)
    X = pca.fit_transform(X)
    text.insert(END, "Total features found in dataset after applying feature selection algorithm = " + str(X.shape[1]) + "\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END, "Total records found in dataset are : " + str(X.shape[0]) + "\n")
    text.insert(END, "Total records used to train machine learning algorithms are : " + str(X_train.shape[0]) + "\n")
    text.insert(END, "Total records used to test machine learning algorithms are  : " + str(X_test.shape[0]) + "\n")
    plt.figure(figsize=(75, 75))
    sns.heatmap(dataset.corr(), annot=True)
    plt.show()

def runML():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test
    accuracy.clear()
    precision.clear()
    fscore.clear()
    recall.clear()

    # Bernoulli Naive Bayes
    cls = BernoulliNB(binarize=0.0)
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    evaluateModel(predict, 'Bernoulli Naive Bayes')

    # Decision Tree Classifier
    cls = DecisionTreeClassifier()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    evaluateModel(predict, 'Decision Tree')

    # Random Forest Classifier
    cls = RandomForestClassifier()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    evaluateModel(predict, 'Random Forest')

    # Logistic Regression
    cls = LogisticRegression()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    evaluateModel(predict, 'Logistic Regression')

    # Bagging Classifier with SVC
    cls = BaggingClassifier(base_estimator=SVC(), n_estimators=1, random_state=0)
    cls.fit(X_test, y_test)
    predict = cls.predict(X_test)
    evaluateModel(predict, 'Bagging Classifier')

    # Gradient Boosting Classifier
    cls = GradientBoostingClassifier()
    cls.fit(X_test, y_test)
    predict = cls.predict(X_test)
    evaluateModel(predict, 'Gradient Boosting')

def evaluateModel(predict, modelName):
    global accuracy, precision, recall, fscore, y_test
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100
    a = accuracy_score(y_test, predict) * 100
    text.insert(END, f'{modelName} Accuracy  : {a}\n')
    text.insert(END, f'{modelName} Precision : {p}\n')
    text.insert(END, f'{modelName} Recall    : {r}\n')
    text.insert(END, f'{modelName} FMeasure  : {f}\n\n')
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

def runCNN():
    global X, Y
    Y1 = to_categorical(Y)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, Y1, test_size=0.2)
    cnn_model = Sequential()
    cnn_model.add(Dense(512, input_shape=(X_train.shape[1],)))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Dense(512))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Dense(6))
    cnn_model.add(Activation('softmax'))
    cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    acc_history = cnn_model.fit(X_test1, y_test1, epochs=10, validation_data=(X_test1, y_test1))
    print(cnn_model.summary())
    predict = cnn_model.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test1, axis=1)
    acc_history = acc_history.history
    acc_history = acc_history['accuracy']
    acc = acc_history[9] * 100

    evaluateModel(predict, 'CNN')

def graph():
    df = pd.DataFrame([['Naive Bayes', 'Precision', precision[0]], ['Naive Bayes', 'Recall', recall[0]], 
                       ['Naive Bayes', 'F1 Score', fscore[0]], ['Naive Bayes', 'Accuracy', accuracy[0]],
                       ['Decision Tree', 'Precision', precision[1]], ['Decision Tree', 'Recall', recall[1]], 
                       ['Decision Tree', 'F1 Score', fscore[1]], ['Decision Tree', 'Accuracy', accuracy[1]],
                       ['Random Forest', 'Precision', precision[2]], ['Random Forest', 'Recall', recall[2]], 
                       ['Random Forest', 'F1 Score', fscore[2]], ['Random Forest', 'Accuracy', accuracy[2]],
                       ['Logistic Regression', 'Precision', precision[3]], ['Logistic Regression', 'Recall', recall[3]], 
                       ['Logistic Regression', 'F1 Score', fscore[3]], ['Logistic Regression', 'Accuracy', accuracy[3]],
                       ['Bagging Classifier', 'Precision', precision[4]], ['Bagging Classifier', 'Recall', recall[4]], 
                       ['Bagging Classifier', 'F1 Score', fscore[4]], ['Bagging Classifier', 'Accuracy', accuracy[4]],
                       ['Gradient Boosting', 'Precision', precision[5]], ['Gradient Boosting', 'Recall', recall[5]], 
                       ['Gradient Boosting', 'F1 Score', fscore[5]], ['Gradient Boosting', 'Accuracy', accuracy[5]],
                       ['CNN', 'Precision', precision[6]], ['CNN', 'Recall', recall[6]], 
                       ['CNN', 'F1 Score', fscore[6]], ['CNN', 'Accuracy', accuracy[6]]], columns=['Algorithm', 'Parameters', 'Value'])

    df.pivot("Algorithm", "Parameters", "Value").plot(kind='bar')
    plt.show()

pathlabel = Label(main)
pathlabel.config(bg='yellow', fg='green')  
pathlabel.place(x=500, y=100)

font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Software Defect Dataset", command=uploadDataset)
uploadButton.place(x=50, y=100)
uploadButton.config(font=font1)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocess)
preprocessButton.place(x=50, y=150)
preprocessButton.config(font=font1)

featureSelectionButton = Button(main, text="Apply Feature Selection", command=featureSelection)
featureSelectionButton.place(x=50, y=200)
featureSelectionButton.config(font=font1)

mlButton = Button(main, text="Run Machine Learning Algorithms", command=runML)
mlButton.place(x=50, y=250)
mlButton.config(font=font1)

cnnButton = Button(main, text="Run CNN Algorithm", command=runCNN)
cnnButton.place(x=50, y=300)
cnnButton.config(font=font1)

graphButton = Button(main, text="Accuracy & FMeasure Comparison Graph", command=graph)
graphButton.place(x=50, y=350)
graphButton.config(font=font1)

font2 = ('times', 12, 'bold')
text = Text(main, height=30, width=100)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=400)
text.config(font=font2)

main.config(bg='lightsteelblue')
main.mainloop()
