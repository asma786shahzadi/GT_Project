import csv
from bs4 import BeautifulSoup
from selenium import webdriver
import math
import time
import re

def auto_scroll(driver):
    z = 1000
    previous_height = -math.inf
    while True:
        z += 1000
        current_height = driver.execute_script("return document.documentElement.scrollHeight")
        if current_height == previous_height:
            break
        previous_height = current_height
        scroll = "window.scrollTo(0," + str(z) + ")"
        driver.execute_script(scroll)
        time.sleep(5)
        z += 1000

def preprocess_text(text):
    # Remove special characters, punctuation, and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    # Convert text to lowercase
    text = text.lower()
    return text

if __name__ == "__main__":
    driver = webdriver.Chrome()

    urls = ['https://www.everydayhealth.com/fitness/best-resistance-band-exercises-to-strengthen-the-abs-and-core/',
            'https://www.everydayhealth.com/fitness/resistance-band-exercises-for-stronger-arms-and-shoulders/',
            'https://www.everydayhealth.com/heart-health/how-isometric-exercise-can-improve-blood-pressure/',
            'https://www.everydayhealth.com/heart-failure/living-with/safe-exercises-people-with-heart-failure/',
            'https://www.everydayhealth.com/hs/ankylosing-spondylitis-treatment-management/pictures/smart-exercises/',
            'https://www.everydayhealth.com/hs/womens-health/exercise-help-endometriosis/',
            'https://www.everydayhealth.com/depression-pictures/great-exercises-to-fight-depression.aspx',
            'https://www.everydayhealth.com/depression/running-walking-yoga-lifting-weights-most-effective-exercises-for-depression/',
            'https://www.everydayhealth.com/sleep-disorders/restless-leg-syndrome/10-ways-exercise-with-restless-legs-syndrome/',
            'https://www.everydayhealth.com/atrial-fibrillation/safe-exercises-when-you-have-a-fib/',
            'https://www.everydayhealth.com/atrial-fibrillation/more-than-one-in-four-women-experience-atrial-fibrillation-after-menopause/',
            'https://www.everydayhealth.com/menopause/study-compares-safety-of-estrogen-patches-pills-and-creams/',
            'https://www.everydayhealth.com/womens-health/hormones/history-hormone-therapy/',
            'https://www.everydayhealth.com/menopause/hormone-therapy-hot-flashes-not-disease-prevention/',
            'https://www.everydayhealth.com/menopause/migraine-attacks-and-hot-flashes-tied-to-heart-risks-after-menopause/',
            'https://www.everydayhealth.com/wellness/what-is-infrared-sauna-therapy-a-complete-guide-for-beginners/',
            'https://www.everydayhealth.com/diet-nutrition/diet/scientific-health-benefits-turmeric-curcumin/',
            ]

    classes = ['eh-content-block__content']

    for url in urls:
        driver.get(url)
        auto_scroll(driver)

        content = driver.page_source
        soup = BeautifulSoup(content, "html.parser")
        titles = soup.find_all('div', class_=classes)

        titles_text = [title.text.strip() for title in titles]

        # Preprocess titles
        preprocessed_titles = [preprocess_text(title) for title in titles_text]

        # Generate filename from URL
        filename = url.split('/')[-2] + '.csv'

        # Write titles to CSV file
        with open(filename, "w", newline='', encoding="utf-8") as csvfile:
            csvwriter = csv.writer(csvfile)
            for title in preprocessed_titles:
                csvwriter.writerow([title])  # Write each title as a separate row

        print("Titles scraped from", url, "saved to", filename)

    driver.quit()
