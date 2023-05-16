# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, \
    f1_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score
import warnings

warnings.filterwarnings("ignore")


np.random.seed(0)

data = pd.read_csv(r"D:\Projects\Research-YT\Research-YT-Final\Process_Audio\New_Hate_Sentiment\fetal_health.csv")
data.head()

data.info()

data.describe().transpose()

data.isnull().sum()

data['fetal_health'].value_counts()

data['fetal_health'].value_counts().plot(kind='bar', color='green')

# correlation matrix
corrmat = data.corr()
plt.figure(figsize=(15, 15))

cmap = sns.diverging_palette(250, 10, s=80, l=55, n=9, as_cmap=True)

sns.heatmap(corrmat, annot=True, cmap=cmap, center=0)

# assigning values to features as X and target as y
X = data.drop(["fetal_health"], axis=1)
y = data["fetal_health"]

# Set up a standard scaler for the features
col_names = list(X.columns)
s_scaler = preprocessing.StandardScaler()
X_df = s_scaler.fit_transform(X)
X_df = pd.DataFrame(X_df, columns=col_names)
X_df.describe().transpose()

# looking at the scaled features
plt.figure(figsize=(20, 10))
sns.boxenplot(data=X_df, orient="h", palette="Set2")
plt.xticks(rotation=90)
plt.show()

# spliting test and training sets
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.3, random_state=42)

# A quick model selection process
# pipelines of models( it is short was to fit and pred)
pipeline_lr = Pipeline([('lr_classifier', LogisticRegression(random_state=42))])

pipeline_dt = Pipeline([('dt_classifier', DecisionTreeClassifier(random_state=42))])

pipeline_rf = Pipeline([('rf_classifier', RandomForestClassifier())])

pipeline_svc = Pipeline([('sv_classifier', SVC())])

# List of all the pipelines
pipelines = [pipeline_lr, pipeline_dt, pipeline_rf, pipeline_svc]

# Dictionary of pipelines and classifier types for ease of reference
pipe_dict = {0: 'Logistic Regression', 1: 'Decision Tree', 2: 'RandomForest', 3: "SVC"}

# Fit the pipelines
for pipe in pipelines:
    pipe.fit(X_train, y_train)


# cross validation on accuracy
cv_results_accuracy = []
for i, model in enumerate(pipelines):
    cv_score = cross_val_score(model, X_train, y_train, cv=10)
    cv_results_accuracy.append(cv_score)
    print("%s: %f " % (pipe_dict[i], cv_score.mean()))

# taking look at the test set
pred_rfc = pipeline_rf.predict(X_test)
accuracy = accuracy_score(y_test, pred_rfc)
print("Accuracy: ", accuracy, "\n")

# Building a dictionalry with list of optional values that will me analyesed by GridSearch CV
parameters = {
    'n_estimators': [100, 150, 200, 500, 700, 900],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 6, 8, 12, 14, 16],
    'criterion': ['gini', 'entropy'],
    'n_jobs': [-1, 1, None]
}

# Fitting the trainingset to find parameters with best accuracy
CV_rfc = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters, cv=5)
CV_rfc.fit(X_train, y_train)

# Getting the outcome of gridsearch

print(CV_rfc.best_params_)

RF_model = RandomForestClassifier(**CV_rfc.best_params_)
RF_model.fit(X_train, y_train)
# Testing the Model on test set
predictions = RF_model.predict(X_test)
acccuracy = accuracy_score(y_test, predictions)
acccuracy

acccuracy = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions, average="weighted")
precision = precision_score(y_test, predictions, average="weighted")
f1_score = f1_score(y_test, predictions, average="micro")

print("********* Random Forest Results *********")
print("Accuracy    : ", acccuracy)
print("Recall      : ", recall)
print("Precision   : ", precision)
print("F1 Score    : ", f1_score)

print(classification_report(y_test, predictions))

# cofusion matrix
plt.subplots(figsize=(12, 8))
cf_matrix = confusion_matrix(y_test, predictions)
sns.heatmap(cf_matrix / np.sum(cf_matrix), cmap=cmap, annot=True, annot_kws={'size': 15})
