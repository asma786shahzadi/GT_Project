
#------- graph plot and mcs 
import os
import nltk
import networkx as nx
import matplotlib.pyplot as plt 
from nltk.tokenize import word_tokenize

# Download NLTK resources if necessary
nltk.download('punkt')

# Function to construct a co-occurrence graph from a text file
def construct_co_occurrence_graph(file_path):
    graph = nx.Graph()
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        tokens = word_tokenize(text)
        # Create nodes
        unique_tokens = set(tokens)  # Filter out duplicate tokens
        graph.add_nodes_from(unique_tokens)
        # Create edges based on co-occurrence (e.g., within a window of 2 words)
        for i in range(len(tokens)):
            for j in range(i+1, min(i+3, len(tokens))):  # Consider a window of 2 words
                if not graph.has_edge(tokens[i], tokens[j]):
                    graph.add_edge(tokens[i], tokens[j], weight=1)
                else:
                    graph[tokens[i]][tokens[j]]['weight'] += 1
    return graph

if __name__ == "__main__":
    input_dir = "preprocessed_data"
    file_paths = [os.path.join(input_dir, file_name) for file_name in os.listdir(input_dir)]
    
    # Construct co-occurrence graphs from text files
    graphs = [construct_co_occurrence_graph(file_path) for file_path in file_paths]

    # Visualize the constructed graphs and compute Maximum Clique Size (MCS)
    for i, graph in enumerate(graphs):
        if i < 12:  # Compute MCS for only the first 12 graphs
            max_clique_size = len(max(nx.find_cliques(graph), key=len))
            print(f"Graph {i+1} - Maximum Clique Size: {max_clique_size}")

        # Plot the graph for all files
        nx.draw(graph, with_labels=True)
        plt.title(f"Graph {i+1}")
        plt.show()





# #------------12 graph 
# import os
# import nltk
# import networkx as nx
# import matplotlib.pyplot as plt
# from nltk.tokenize import word_tokenize
# from sklearn.neighbors import kneighbors_graph

# # Download NLTK resources if necessary
# nltk.download('punkt')

# # Function to construct a k-nearest neighbor graph from a text file
# def construct_knn_graph(file_path, k=5):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         text = file.read()
#         tokens = word_tokenize(text)
#         # Create nodes
#         unique_tokens = set(tokens)  # Filter out duplicate tokens
#         graph = nx.Graph()
#         graph.add_nodes_from(unique_tokens)
#         # Compute pairwise distances
#         X = [[tokens.index(token)] for token in unique_tokens]  # Convert tokens to indices
#         # Build k-nearest neighbor graph
#         knn_graph = kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=False)
#         # Add edges to the graph based on k-nearest neighbors
#         for i, row in enumerate(knn_graph):
#             neighbors = row.indices
#             for neighbor in neighbors:
#                 if i != neighbor:  # Exclude self-loops
#                     graph.add_edge(list(unique_tokens)[i], list(unique_tokens)[neighbor])
#     return graph

# if __name__ == "__main__":
#     input_dir = "preprocessed_data"
#     file_paths = [os.path.join(input_dir, file_name) for file_name in os.listdir(input_dir)]
    
#     # Construct k-nearest neighbor graphs from text files and compute MCS for the first 12 graphs
#     for i, file_path in enumerate(file_paths[:12]):
#         graph = construct_knn_graph(file_path)
#         max_clique_size = len(max(nx.find_cliques(graph), key=len))
#         print(f"Graph {i+1} - Maximum Clique Size: {max_clique_size}")

#         # Plot the graph
#         nx.draw(graph, with_labels=True)
#         plt.title(f"Graph {i+1}")
#         plt.show()



# import os
# import nltk
# import networkx as nx
# import matplotlib.pyplot as plt
# from nltk.tokenize import word_tokenize
# from sklearn.neighbors import kneighbors_graph

# # Download NLTK resources if necessary
# nltk.download('punkt')

# # Function to construct a k-nearest neighbor graph from a text file
# def construct_knn_graph(file_path, k=5):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         text = file.read()
#         tokens = word_tokenize(text)
#         # Create nodes
#         unique_tokens = set(tokens)  # Filter out duplicate tokens
#         graph = nx.Graph()
#         graph.add_nodes_from(unique_tokens)
#         # Compute pairwise distances
#         X = [[tokens.index(token)] for token in unique_tokens]  # Convert tokens to indices
#         # Build k-nearest neighbor graph
#         knn_graph = kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=False)
#         # Add edges to the graph based on k-nearest neighbors
#         for i, row in enumerate(knn_graph):
#             neighbors = row.indices
#             for neighbor in neighbors:
#                 if i != neighbor:  # Exclude self-loops
#                     graph.add_edge(list(unique_tokens)[i], list(unique_tokens)[neighbor])
#     return graph

# if __name__ == "__main__":
#     input_dir = "preprocessed_data"
#     file_paths = [os.path.join(input_dir, file_name) for file_name in os.listdir(input_dir)]
    
#     # Construct k-nearest neighbor graphs from text files and compute MCS for the first 12 graphs
#     for i, file_path in enumerate(file_paths[:12]):
#         graph = construct_knn_graph(file_path)
#         max_clique_size = len(max(nx.find_cliques(graph), key=len))
#         print(f"Graph {i+1} - Maximum Clique Size: {max_clique_size}")

#         # Plot the graph in a separate figure window
#         plt.figure()
#         nx.draw(graph, with_labels=True)
#         plt.title(f"Graph {i+1}")
#         plt.show()

