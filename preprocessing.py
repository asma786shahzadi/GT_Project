import re

def preprocess_text(text):
    # Remove special characters and extra whitespaces
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    # Convert text to lowercase
    text = text.lower()
    return text.strip()

def preprocess_txt(input_filename, output_filename):
    with open(input_filename, 'r', encoding='utf-8') as infile:
        text = infile.read()

    cleaned_text = preprocess_text(text)

    with open(output_filename, 'w', encoding='utf-8') as outfile:
        outfile.write(cleaned_text)

if __name__ == "__main__":
    urls = [
          'https://health.ucdavis.edu/blog/cultivating-health/category/exercise-and-fitness',
           'https://health.ucdavis.edu/blog/cultivating-health/category/children-health',
           'https://health.ucdavis.edu/blog/cultivating-health/category/exercise-and-fitness',
           'https://health.ucdavis.edu/blog/cultivating-health/category/mental-health',
           'https://health.ucdavis.edu/blog/cultivating-health/archive',
           'https://health.ucdavis.edu/blog/cultivating-health/category/cancer',
           'https://health.ucdavis.edu/blog/cultivating-health'
    ]

    for index, url in enumerate(urls):
        input_filename = f"data_{index}.txt"
        output_filename = f"cleaned_data_{index}.txt"
        preprocess_txt(input_filename, output_filename)
        print("Preprocessed data from", input_filename, "saved to", output_filename)

#--------------for txt files 
# import os
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer
# from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# def preprocess_text(text):
#     # Tokenization
#     tokens = word_tokenize(text.lower())
    
#     # Remove stop words
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if word not in stop_words]
    
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
#     # Stemming (Optional)
#     # stemmer = PorterStemmer()
#     # tokens = [stemmer.stem(word) for word in tokens]
    
#     return " ".join(tokens)

# def preprocess_file(input_file, output_file):
#     with open(input_file, 'r', encoding='utf-8') as file:
#         text = file.read()
#         preprocessed_text = preprocess_text(text)
#         with open(output_file, 'w', encoding='utf-8') as out_file:
#             out_file.write(preprocessed_text)

# if __name__ == "__main__":
#     input_dir = "E:/6th semester\GT\scrapping_fashion&beauty"
#     output_dir = "preprocessed_data"

#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     for file_name in os.listdir(input_dir):
#         input_file_path = os.path.join(input_dir, file_name)
#         output_file_path = os.path.join(output_dir, file_name)
#         preprocess_file(input_file_path, output_file_path)
#         print(f"Preprocessed {file_name} saved to {output_file_path}")

# #--------------for txt files 
# import os
# import re

# def preprocess_text(text):
#     # Remove special characters and extra whitespaces
#     text = re.sub(r'\W+', ' ', text)
#     text = re.sub(r'\s+', ' ', text)
#     # Convert text to lowercase
#     text = text.lower()
#     return text.strip()

# def preprocess_txt(input_filename, output_filename):
#     with open(input_filename, 'r', encoding='utf-8') as infile:
#         text = infile.read()

#     cleaned_text = preprocess_text(text)

#     with open(output_filename, 'w', encoding='utf-8') as outfile:
#         outfile.write(cleaned_text)

# if __name__ == "__main__":
#     input_dir = "E:/6th semester/GT/scrapping_fashion&beauty"
#     output_dir = r"E:/6th semester/GT/scrapping_fashion&beauty/cleaned_data"
    
#     # Ensure output directory exists
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     file_names = os.listdir(input_dir)
    
#     for index, file_name in enumerate(file_names):
#         input_filename = os.path.join(input_dir, file_name)
#         output_filename = os.path.join(output_dir, f"cleaned_data_{index}.txt")
#         preprocess_txt(input_filename, output_filename)
#         print("Preprocessed data from", input_filename, "saved to", output_filename)
