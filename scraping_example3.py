

from bs4 import BeautifulSoup
from selenium import webdriver
import math
import time

def AutoScroll(driver):
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


if __name__ == "__main__":
    driver = webdriver.Chrome()

    urls =['https://www.everydayhealth.com/wellness/',
           'https://www.everydayhealth.com/lifestyle/food/',
           'https://www.everydayhealth.com/drugs/',
           'https://www.everydayhealth.com/emotional-health/all-articles/',
           'https://www.everydayhealth.com/sexual-health/sexually-transmitted-diseases/',
           'https://www.everydayhealth.com/diet-nutrition/the-dash-diet.aspx',
           'https://www.everydayhealth.com/wellness/healthy-skin/',
           'https://www.everydayhealth.com/self-care/']
            

    for index, url in enumerate(urls):
        driver.get(url)
        AutoScroll(driver)

        content = driver.page_source
        soup = BeautifulSoup(content, "html.parser")
        titles = soup.find_all('div', class_='landing-content-list__item-dek')
        titles_text = [title.text.strip() for title in titles]

        filename = f"data_{index}.txt"
        with open(filename, "w", encoding="utf-8") as txtfile:
            for title in titles_text:
                txtfile.write(title + "\n")

        print("Titles scraped from", url, "saved to", filename)

    driver.quit()
