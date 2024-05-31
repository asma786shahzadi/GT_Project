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
       'https://www.everydayhealth.com/wellness/',
           'https://www.everydayhealth.com/lifestyle/food/',
           'https://www.everydayhealth.com/drugs/',
           'https://www.everydayhealth.com/emotional-health/all-articles/',
           'https://www.everydayhealth.com/sexual-health/sexually-transmitted-diseases/',
           'https://www.everydayhealth.com/diet-nutrition/the-dash-diet.aspx',
           'https://www.everydayhealth.com/wellness/healthy-skin/',
           'https://www.everydayhealth.com/self-care/'
    ]

    for index, url in enumerate(urls):
        input_filename = f"data_{index}.txt"
        output_filename = f"cleaned_data_{index}.txt"
        preprocess_txt(input_filename, output_filename)
        print("Preprocessed data from", input_filename, "saved to", output_filename)
