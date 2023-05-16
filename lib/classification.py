import pickle
import os

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def classify(df):
    # Split the data into training and testing sets
    global best_model, best_model_name
    X = df.drop('target', axis=1)  # drop the target column
    y = df['target']

    # scale the data
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Create the logistic regression model
    lr = LogisticRegression()
    sgd = SGDClassifier()
    svc = SVC(probability=True)
    rf = RandomForestClassifier()
    nb = GaussianNB()
    dt = DecisionTreeClassifier()
    gb = GradientBoostingClassifier()

    # Fit the model to the training data
    lr.fit(X_train, y_train)
    sgd.fit(X_train, y_train)
    svc.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    nb.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    # Use the trained model to make predictions on the test data
    y_pred = lr.predict(X_test)
    y_pred_sgd = sgd.predict(X_test)
    y_pred_svc = svc.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    y_pred_nb = nb.predict(X_test)
    y_pred_dt = dt.predict(X_test)
    y_pred_gb = gb.predict(X_test)

    # Evaluate the accuracy of the model
    lr_accuracy = accuracy_score(y_test, y_pred)
    sgd_accuracy = accuracy_score(y_test, y_pred_sgd)
    svc_accuracy = accuracy_score(y_test, y_pred_svc)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    nb_accuracy = accuracy_score(y_test, y_pred_nb)
    dt_accuracy = accuracy_score(y_test, y_pred_dt)
    gb_accuracy = accuracy_score(y_test, y_pred_gb)

    print('Accuracy lr: {:.2f}%'.format(lr_accuracy * 100))
    print('Accuracy sgd: {:.2f}%'.format(sgd_accuracy * 100))
    print('Accuracy svc: {:.2f}%'.format(svc_accuracy * 100))
    print('Accuracy rf: {:.2f}%'.format(rf_accuracy * 100))
    print('Accuracy nb: {:.2f}%'.format(nb_accuracy * 100))
    print('Accuracy dt: {:.2f}%'.format(dt_accuracy * 100))
    print('Accuracy gb: {:.2f}%'.format(gb_accuracy * 100))

    # if log.txt exists delete it
    if os.path.exists('log.txt'):
        os.remove('log.txt')

    # create log.txt file and write the accuracies
    with open('log.txt', 'w') as f:
        f.write("MIDI Evaluator Training Log\n")
        f.write("=====================================\n\n")
        f.write('Accuracy lr: {:.2f}%'.format(lr_accuracy * 100))
        f.write('\nAccuracy sgd: {:.2f}%'.format(sgd_accuracy * 100))
        f.write('\nAccuracy svc: {:.2f}%'.format(svc_accuracy * 100))
        f.write('\nAccuracy rf: {:.2f}%'.format(rf_accuracy * 100))
        f.write('\nAccuracy nb: {:.2f}%'.format(nb_accuracy * 100))
        f.write('\nAccuracy dt: {:.2f}%'.format(dt_accuracy * 100))
        f.write('\nAccuracy gb: {:.2f}%'.format(gb_accuracy * 100))
        f.write('\n')

    # get max accuracy
    max_accuracy = max(lr_accuracy, sgd_accuracy, svc_accuracy, rf_accuracy, dt_accuracy, gb_accuracy)

    # open log.txt file and write the best model with its accuracy
    with open('log.txt', 'a') as f:
        if lr_accuracy == max_accuracy:
            f.write('\n\nBest Model: Logistic Regression  -  ')
            f.write('Accuracy: {:.2f}%'.format(lr_accuracy * 100))
        elif sgd_accuracy == max_accuracy:
            f.write('\n\nBest Model: SGD Classifier  -  ')
            f.write('Accuracy: {:.2f}%'.format(sgd_accuracy * 100))
        elif svc_accuracy == max_accuracy:
            f.write('\n\nBest Model: Support Vector Classifier  -  ')
            f.write('Accuracy: {:.2f}%'.format(svc_accuracy * 100))
        elif rf_accuracy == max_accuracy:
            f.write('\n\nBest Model: Random Forest Classifier  -  ')
            f.write('Accuracy: {:.2f}%'.format(rf_accuracy * 100))
        elif nb_accuracy == max_accuracy:
            f.write('\n\nBest Model: Naive Bayes Classifier  -  ')
            f.write('Accuracy: {:.2f}%'.format(nb_accuracy * 100))
        elif dt_accuracy == max_accuracy:
            f.write('\n\nBest Model: Decision Tree Classifier  -  ')
            f.write('Accuracy: {:.2f}%'.format(dt_accuracy * 100))
        elif gb_accuracy == max_accuracy:
            f.write('\n\nBest Model: Gradient Boosting Classifier  -  ')
            f.write('Accuracy: {:.2f}%'.format(gb_accuracy * 100))

    # print the best model
    if lr_accuracy == max_accuracy:
        print("Logistic Regression")
        best_model = lr
        best_model_name = "Logistic Regression"
    elif sgd_accuracy == max_accuracy:
        print("SGD Classifier")
        best_model = sgd
        best_model_name = "SGD Classifier"
    elif svc_accuracy == max_accuracy:
        print("Support Vector Classifier")
        best_model = svc
        best_model_name = "Support Vector Classifier"
    elif rf_accuracy == max_accuracy:
        print("Random Forest Classifier")
        best_model = rf
        best_model_name = "Random Forest Classifier"
    elif nb_accuracy == max_accuracy:
        print("Naive Bayes Classifier")
        best_model = nb
        best_model_name = "Naive Bayes Classifier"
    elif dt_accuracy == max_accuracy:
        print("Decision Tree Classifier")
        best_model = dt
        best_model_name = "Decision Tree Classifier"
    elif gb_accuracy == max_accuracy:
        print("Gradient Boosting Classifier")
        best_model = gb
        best_model_name = "Gradient Boosting Classifier"

    # save the best model
    pickle.dump(best_model, open('models/model.pkl', 'wb'))

    # save scaler
    pickle.dump(scaler, open('models/scaler.pkl', 'wb'))

    return best_model, scaler, best_model_name



def cluster_model(df):
    # add target column
    df['target'] = 0

    # drop columns withn data type object
    df = df.drop(df.select_dtypes(include=['object']), axis=1)

    # Split the data into training and testing sets
    X = df.drop('target', axis=1)  # drop the target column

    print(X)

    # Scale the data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cluster the data
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X_scaled)

    # Assign clusters to data points
    clusters = kmeans.predict(X_scaled)

    # Add cluster labels to the dataframe
    df['target'] = clusters

    # Save the dataframe
    df.to_csv("clustered_data.csv", index=False)

    return df


def classification_predict(df):
    # load the model
    model = pickle.load(open('models/model.pkl', 'rb'))

    # load the scaler
    scaler = pickle.load(open('models/scaler.pkl', 'rb'))

    # remove null values
    df = df.dropna()

    # drop duplicate rows
    df = df.drop_duplicates()

    # for each row in the dataframe predict
    for index, row in df.iterrows():
        # get the features
        features = row.values[1:-1]

        # scale the features
        features = scaler.transform([features])

        # predict
        prediction = model.predict(features)

        # add the prediction to the dataframe
        df.loc[index, 'severity'] = prediction[0]

    # save the dataframe
    df.to_csv("predicted_data.csv", index=False)

    return df



