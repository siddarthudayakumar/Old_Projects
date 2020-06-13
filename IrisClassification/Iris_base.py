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

# get stats for the data
print(iris_dataset.describe())
print('--------------------------------')
print(iris_dataset.groupby('class').size())


