import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
from matplotlib import cm
from pandas.plotting import scatter_matrix
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

# read table deprecated in latest version
print(sklearn.__version__)
fruits = pd.read_csv('./data.txt', sep='\t')

# Count-plot
sns.countplot(fruits['fruit_name'], label="Count")
plt.savefig('plots/count_plot')
# plt.show()

# Box-plot
fruits.drop('fruit_label', axis=1).plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False,
                                        figsize=(9, 9), title='Box Plot')
plt.savefig('plots/fruit_box')
# plt.show()

# Histogram
fruits.drop('fruit_label', axis=1).hist(bins=30, figsize=(9, 9))
pl.suptitle("Histogram for numeric input variables")
plt.savefig('plots/fruits_hist')
# plt.show()

# Scatter Plot
feature_names = ['mass', 'width', 'height', 'color_score']
X = fruits[feature_names]
Y = fruits['fruit_label']
c_map = cm.get_cmap('gnuplot')
scatter = scatter_matrix(X, c=Y, marker='o', s=40, hist_kwds={'bins': 15}, figsize=(9, 9), cmap=c_map)
plt.suptitle('Scatter Matrix for input variables')
plt.savefig('plots/fruits_scatter')
# plt.show()

# Statistical Summary
summary = pd.DataFrame.describe(fruits)

# Scaling the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
scale = MinMaxScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

# Logistic Regression
log_reg = LogisticRegression(solver='lbfgs', multi_class='auto')
log_reg.fit(X_train, Y_train)
accuracy_train_log_reg = log_reg.score(X_train, Y_train)  # 0.46
accuracy_test_log_reg = log_reg.score(X_test, Y_test)  # 0.75

# Decision Tree

clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)
accuracy_train_dec_tree = clf.score(X_train, Y_train)  # 1.00
accuracy_test_dec_tree = clf.score(X_test, Y_test)  # 0.733

# K-Nearest Neighbours
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
accuracy_train_knn = knn.score(X_train, Y_train)  # 0.95
accuracy_test_knn = knn.score(X_test, Y_test)  # 1.0

# Linear Discriminant Analysis

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
accuracy_train_lda = lda.score(X_train, Y_train)  # 0.86
accuracy_test_lda = lda.score(X_test, Y_test)  # 0.67

# Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
# accuracy_train_gnb = gnb.score(X_train, X_test)
# accuracy_test_gnb = gnb.score(X_test, Y_test)
