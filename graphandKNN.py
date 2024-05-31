
import os
import nltk
import networkx as nx
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Download NLTK resources if necessary
nltk.download('punkt')
nltk.download('stopwords')

# Define global variables
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
    return tokens

# Function to construct graph from text
def construct_graph(text):
    tokens = preprocess_text(text)
    graph = nx.DiGraph()
    for i in range(len(tokens) - 1):
        word1, word2 = tokens[i], tokens[i+1]
        if not graph.has_edge(word1, word2):
            graph.add_edge(word1, word2, weight=1)
        else:
            graph[word1][word2]['weight'] += 1
    return graph

folder_path1 = r'D:\Semester 6\Graph theory\Project\fashion&beauty\preprocessed_data'
folder_path2=  r'D:\Semester 6\Graph theory\Project\Health&fitness\preprocessed_data'
folder_path3= r'D:\Semester 6\Graph theory\Project\Business&Finance\preprocess_data'
# Load data for each topic
def load_data(folder_path):
    data = []
    for i in range(1, 13):  # Loop from 1 to 12
        filename = f"cleaned_data_{i}.txt"  # Add file extension '.txt'
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
            text = file.read()
            data.append(text)
    return data

# Function to preprocess and construct graphs for training data
def preprocess_and_construct_graphs(data):
    graphs = []
    for text in data:
        graph = construct_graph(text)
        graphs.append(graph)
    return graphs

# Function to extract features from graphs
def extract_features(graphs):
    features = []
    for graph in graphs:
        # Extract features from the graph (example feature: number of nodes)
        features.append(len(graph.nodes))
    return features

if __name__ == "__main__":
    # Data Collection and Preparation
    topic1_data = load_data(folder_path1)
    topic2_data = load_data(folder_path2)
    topic3_data = load_data(folder_path3)
    # Construct graphs for training data
    topic1_graphs = preprocess_and_construct_graphs(topic1_data)
    topic2_graphs = preprocess_and_construct_graphs(topic2_data)
    topic3_graphs = preprocess_and_construct_graphs(topic3_data)
    # Extract features from graphs
    topic1_features = extract_features(topic1_graphs)
    topic2_features = extract_features(topic2_graphs)
    topic3_features = extract_features(topic3_graphs)

    print("Length of topic1_features:", len(topic1_features))
    print("Length of topic2_features:", len(topic2_features))
    print("Length of topic2_features:", len(topic3_features))
    # Prepare training data and labels
    X_train = topic1_features[:12] + topic2_features[:12] + topic3_features[:12]
    y_train = [1] * 12 + [2] * 12 + [3] * 12

    # Prepare test data and labels
    X_test = topic1_features[:12] + topic2_features[:12] + topic3_features[:12]
    y_test = [1] * 12 + [2] * 12 + [3] *12

    # Check if test data is not empty
    if len(X_test) == 0:
        print("Error: Test data is empty.")
    else:
        # Reshape X_train
        X_train_reshaped = np.array(X_train).reshape(-1, 1)

        # Handle NaN values using SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X_train_imputed = imputer.fit_transform(X_train_reshaped)

        # Reshape and impute test data only if it's not empty
        X_test_imputed = imputer.transform(np.array(X_test).reshape(-1, 1))

        # Train the KNN model
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(X_train_imputed, y_train)

        # Predict labels for the test set
        y_pred = knn_model.predict(X_test_imputed)

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

        # Classification report
        print(classification_report(y_test, y_pred))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
