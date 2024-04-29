# import pandas as pd
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from bs4 import BeautifulSoup
# import requests
# import time
# import networkx as nx
# import matplotlib.pyplot as plt
# import math
# import re

# def auto_scroll(driver):
#     z = 1000
#     previous_height = -math.inf
#     while True:
#         z += 1000
#         current_height = driver.execute_script("return document.documentElement.scrollHeight")
#         if current_height == previous_height:
#             break
#         previous_height = current_height
#         scroll = "window.scrollTo(0," + str(z) + ")"
#         driver.execute_script(scroll)
#         time.sleep(5)
#         z += 1000

# def preprocess_text(text):
#     # Remove HTML tags
#     text = BeautifulSoup(text, 'html.parser').get_text()
#     # Remove special characters, punctuation, and extra whitespace
#     text = re.sub(r'[^\w\s]', ' ', text)
#     text = re.sub(r'\s+', ' ', text)
#     # Convert text to lowercase
#     text = text.lower()
#     # Remove verbs and extra words
#     verbs = ['is', 'are', 'was', 'were', 'am', 'be', 'been', 'being']
#     extra_words = ['the', 'and', 'or', 'in', 'on', 'at', 'to', 'a', 'an', 'as', 'for', 'of']
#     text_tokens = text.split()
#     filtered_tokens = [token for token in text_tokens if token not in verbs and token not in extra_words]
#     text = ' '.join(filtered_tokens)
#     return text

# def scrape_data_from_url(url):
#     # Add headers
#     headers = {'User-Agent': 'Your User Agent String'}
#     try:
#         response = requests.get(url, headers=headers)
#         response.raise_for_status()  # Raise an exception for HTTP errors
#         time.sleep(1)  # Add a 1-second delay
#         # Extract text data from response
#         soup = BeautifulSoup(response.content, 'html.parser')
#         paragraphs = soup.find_all('p')
#         data = ' '.join([p.text.strip() for p in paragraphs])
#         return data
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching data from URL {url}: {e}")
#         return None

# def construct_graph(data):
#     graph = nx.DiGraph()
#     for document in data:
#         terms = document.split()  # Tokenize the document
#         for i in range(len(terms) - 1):
#             term1, term2 = terms[i], terms[i + 1]
#             graph.add_edge(term1, term2)
#     return graph

# if __name__ == "__main__":
#     urls = ['https://www.everydayhealth.com/fitness/best-resistance-band-exercises-to-strengthen-the-abs-and-core/',
#             'https://www.everydayhealth.com/fitness/resistance-band-exercises-for-stronger-arms-and-shoulders/',
#             'https://www.everydayhealth.com/heart-health/how-isometric-exercise-can-improve-blood-pressure/',
#             'https://www.everydayhealth.com/heart-failure/living-with/safe-exercises-people-with-heart-failure/',
#             'https://www.everydayhealth.com/hs/ankylosing-spondylitis-treatment-management/pictures/smart-exercises/',
#             'https://www.everydayhealth.com/hs/womens-health/exercise-help-endometriosis/',
#             'https://www.everydayhealth.com/depression-pictures/great-exercises-to-fight-depression.aspx',
#             'https://www.everydayhealth.com/depression/running-walking-yoga-lifting-weights-most-effective-exercises-for-depression/',
#             'https://www.everydayhealth.com/sleep-disorders/restless-leg-syndrome/10-ways-exercise-with-restless-legs-syndrome/',
#             'https://www.everydayhealth.com/atrial-fibrillation/safe-exercises-when-you-have-a-fib/',
#             'https://www.everydayhealth.com/atrial-fibrillation/more-than-one-in-four-women-experience-atrial-fibrillation-after-menopause/',
#             'https://www.everydayhealth.com/menopause/study-compares-safety-of-estrogen-patches-pills-and-creams/',
#             'https://www.everydayhealth.com/womens-health/hormones/history-hormone-therapy/',
#             'https://www.everydayhealth.com/menopause/hormone-therapy-hot-flashes-not-disease-prevention/',
#             'https://www.everydayhealth.com/menopause/migraine-attacks-and-hot-flashes-tied-to-heart-risks-after-menopause/',
#             'https://www.everydayhealth.com/wellness/what-is-infrared-sauna-therapy-a-complete-guide-for-beginners/',
#             'https://www.everydayhealth.com/diet-nutrition/diet/scientific-health-benefits-turmeric-curcumin/'
#             ]

#     data = []
#     for i, url in enumerate(urls):
#         text_data = scrape_data_from_url(url)
#         if text_data:
#             # Preprocess text data
#             preprocessed_data = preprocess_text(text_data)
#             data.append(preprocessed_data)

#             print(f"Scraped and preprocessed data from URL {url}: {preprocessed_data[:50]}...")  # Print first 50 characters of data
#             print(f"Length of data after scraping and preprocessing URL {url}: {len(data)}")

#             # Construct graph
#             graph = construct_graph([preprocessed_data])

#             # Compute and print graph measures
#             print(f"Graph measures for URL {url}:")
#             print("Number of nodes:", nx.number_of_nodes(graph))
#             print("Number of edges:", nx.number_of_edges(graph))
#             try:
#                 print("Average shortest path length:", nx.average_shortest_path_length(graph))
#             except nx.NetworkXError:
#                 print("Average shortest path length: Graph is not strongly connected.")
#             print("")

#             # Draw and display the graph
#             plt.figure(figsize=(8, 6))
#             nx.draw(graph, with_labels=True)
#             plt.title(f"Graph for URL {url}")
#             plt.show()

#     # Convert text data into TF-IDF vectors
#     vectorizer = TfidfVectorizer()
#     data_tfidf = vectorizer.fit_transform(data)

#     # Prepare labels
#     labels = np.arange(len(urls))

#     # Print lengths of data and labels
#     print("Length of TF-IDF data:", len(data_tfidf.toarray()))
#     print("Length of labels array:", len(labels))

#     # Split data into train and test sets
#     X_train, X_test, y_train, y_test = train_test_split(data_tfidf, labels, test_size=0.2, random_state=42)

#     # Train KNN model
#     knn = KNeighborsClassifier(n_neighbors=3)
#     knn.fit(X_train, y_train)

#     # Predict on test data
#     y_pred = knn.predict(X_test)

#     # Evaluate the model
#     accuracy = accuracy_score(y_test, y_pred)
#     print("Accuracy:", accuracy)



import numpy as np
import networkx as nx
from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Function to construct directed graphs for each document
def construct_graphs(documents):
    graphs = []
    for document in documents:
        terms = document.split()
        graph = nx.DiGraph()
        for i in range(len(terms) - 1):
            term1, term2 = terms[i], terms[i + 1]
            graph.add_edge(term1, term2)
        graphs.append(graph)
    return graphs

# Function to mine frequent subgraphs
def mine_frequent_subgraphs(graphs, min_support=0.5):
    all_edges = []
    for graph in graphs:
        all_edges.extend(graph.edges())

    unique_edges = set(all_edges)
    frequent_subgraphs = []
    for size in range(2, 4):  # Consider subgraphs of size 2 and 3
        for subset in combinations(unique_edges, size):
            support = sum(1 for g in graphs if nx.DiGraph(g.subgraph(subset)).is_isomorphic(g)) / len(graphs)
            if support >= min_support:
                frequent_subgraphs.append(subset)
    return frequent_subgraphs

# Function to extract features from documents based on frequent subgraphs
def extract_features(documents, frequent_subgraphs):
    features = []
    for document in documents:
        document_graph = nx.DiGraph()
        terms = document.split()
        for i in range(len(terms) - 1):
            term1, term2 = terms[i], terms[i + 1]
            if (term1, term2) in frequent_subgraphs:
                document_graph.add_edge(term1, term2)
        features.append(document_graph)
    return features

# Function to compute the maximal common subgraph (MCS) similarity between two graphs
def mcs_similarity(graph1, graph2):
    mcs = nx.algorithms.isomorphism.GraphMatcher(graph1, graph2)
    return len(max(list(mcs.subgraph_isomorphisms_iter()), key=len))

# Function to classify documents using KNN algorithm
def classify_documents(X_train, y_train, X_test, k):
    knn = KNeighborsClassifier(n_neighbors=k, metric=mcs_similarity)
    knn.fit(X_train, y_train)
    return knn.predict(X_test)

# Function to evaluate classification performance
def evaluate_performance(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    return accuracy, precision, recall, f1, cm

if __name__ == "__main__":
    # Dummy data for demonstration
    topics = ['Topic A', 'Topic B', 'Topic C']
    documents_per_topic = 15
    train_documents_per_topic = 12
    test_documents_per_topic = 3

    # Constructing dummy documents
    documents = []
    for topic in topics:
        for _ in range(documents_per_topic):
            documents.append(" ".join([topic] * 300))

    # Splitting data into train and test sets
    train_documents = documents[:train_documents_per_topic] * len(topics)
    test_documents = documents[train_documents_per_topic:] * len(topics)
    train_labels = np.repeat(np.arange(len(topics)), train_documents_per_topic)
    test_labels = np.repeat(np.arange(len(topics)), test_documents_per_topic)

    # Constructing graphs for documents
    train_graphs = construct_graphs(train_documents)
    test_graphs = construct_graphs(test_documents)

    # Mining frequent subgraphs
    frequent_subgraphs = mine_frequent_subgraphs(train_graphs)

    # Extracting features from documents
    train_features = extract_features(train_documents, frequent_subgraphs)
    test_features = extract_features(test_documents, frequent_subgraphs)

    # Classifying test documents
    k = 3  # Number of neighbors for KNN
    y_pred = classify_documents(train_features, train_labels, test_features, k)

    # Evaluating performance
    accuracy, precision, recall, f1, cm = evaluate_performance(test_labels, y_pred)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("Confusion Matrix:")
    print(cm)

    # Plotting confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(np.arange(len(topics)), topics)
    plt.yticks(np.arange(len(topics)), topics)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
