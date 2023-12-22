# Importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.metrics import precision_score, recall_score

# Creating a sidebar and a title 
st.title("Binary Classification Web App")
st.sidebar.title("Binary Classification Webb App")
st.markdown("Are your mushrooms edible or poisonous")
st.sidebar.markdown("Are your mushrooms edible or poisonous")



# A function to load data
@st.cache_data(persist=True) #This will save the results for re-run 
def load_data():
    data = pd.read_csv("mushrooms.csv")
    label = LabelEncoder()
    for col in data.columns:
        data[col] = label.fit_transform(data[col])
    return data

# A function to split our feature and target columns
st.cache_data(persist=True)
def split_data(df):
    X = df.drop(columns='type')
    y = df['type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test

# A function for our plots
def plot_metrics(metrics_list):
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        st.pyplot(fig)
    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots()
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        plt.plot(fpr, tpr, label="ROC Curve")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        st.pyplot(fig)
    if 'Precision-Recall Curve' in metrics_list:
        st.subheader("Precision-Recall Curve")
        fig, ax = plt.subplots()
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        plt.plot(recall, precision, label="Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        st.pyplot(fig)




# Returning our functions
df = load_data()
X_train, X_test, y_train, y_test = split_data(df)

# Choose classifier
st.sidebar.subheader("Choose Classifier")
classifier = st.sidebar.selectbox("Classifier:", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest Classifier"))
if classifier == "Support Vector Machine (SVM)":
    st.sidebar.subheader("Choose Model Hyperparameters")
    C = st.sidebar.number_input("C (Regularization Parameter)", 0.01, 10.0, step=0.01, key='C')
    kernel = st.sidebar.radio("Kernel", ('rbf', 'linear'), key='kernal')
    gamma = st.sidebar.radio("Gamma (Kernel) Coefficient", ('scale', 'auto'), key='gamma')

    metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'), key='metrics')

    if st.sidebar.button("CLASSIFY", key='classify'):
        st.subheader("Support Vector Machine Results:")
        model = SVC(C=C, kernel=kernel, gamma=gamma)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        st.write('Accuracy: ', accuracy.round(2))
        st.write('Precision: ', precision_score(y_test, y_pred, labels=['edible, poisonous']).round(2))
        st.write('Recall: ', recall_score(y_test, y_pred, labels=['edible, poisonous']).round(2))
        plot_metrics(metrics)

if classifier == "Logistic Regression":
    st.sidebar.subheader("Choose Model Hyperparameters")
    C = st.sidebar.number_input("C (Regularization Parameter)", 0.01, 10.0, step=0.01, key='C_LR')
    max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')

    metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'), key='metrics')

    if st.sidebar.button("CLASSIFY", key='classify'):
        st.subheader("Logistic Regression Results:")
        model = LogisticRegression(C=C, max_iter=max_iter)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        st.write('Accuracy: ', accuracy.round(2))
        st.write('Precision: ', precision_score(y_test, y_pred, labels=['edible, poisonous']).round(2))
        st.write('Recall: ', recall_score(y_test, y_pred, labels=['edible, poisonous']).round(2))
        plot_metrics(metrics)

if classifier == "Random Forest Classifier":
    st.sidebar.subheader("Choose Model Hyperparameters")
    n_estimators = st.sidebar.number_input("Number of trees in the forest", 100, 5000, step=100, key='n_estimators')
    max_depth = st.sidebar.number_input("Maximum depth of tree", 1, 20, step=1, key='max_depth')

    metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'), key='metrics')

    if st.sidebar.button("CLASSIFY", key='classify'):
        st.subheader("Random Forest Classifier Results:")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        st.write('Accuracy: ', accuracy.round(2))
        st.write('Precision: ', precision_score(y_test, y_pred, labels=['edible, poisonous']).round(2))
        st.write('Recall: ', recall_score(y_test, y_pred, labels=['edible, poisonous']).round(2))
        plot_metrics(metrics)



# Raw data checkbox
if st.sidebar.checkbox("Raw Data ", False):
    st.subheader("Mushrooms DataSet:")
    st.write(df)
