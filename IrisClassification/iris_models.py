# importing Pandas libraries 
from pandas import read_csv
from pandas.plotting import scatter_matrix

# importing scikit libraries 
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# importing graphing libraries
from matplotlib import pyplot

# Loading the IRIS dataset 
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] # header names
iris_dataset = read_csv(url, names=names)
# print(iris_dataset.head(5))


iris_array = iris_dataset.values
X = iris_array[:,0:4]
y = iris_array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size = 0.10, random_state = 1)

# Creating a list of all the possible classifiers
ml_models = []
ml_models.append(('LR', LogisticRegression(solver = 'liblinear', multi_class = 'ovr')))
ml_models.append(('LDA', LinearDiscriminantAnalysis()))
ml_models.append(('KNN', KNeighborsClassifier()))
ml_models.append(('CART', DecisionTreeClassifier()))
ml_models.append(('NB', GaussianNB()))
ml_models.append(('SVM', SVC(gamma = 'auto')))

# Evaluating all the classifiers. 
results = []
names = []
for name, model in ml_models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean()}")

# Comparing all classifiers
pyplot.boxplot(results, labels = names)
pyplot.show()