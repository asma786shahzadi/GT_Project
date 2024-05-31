# # import csv
# # from bs4 import BeautifulSoup
# # from selenium import webdriver
# # import math
# # import time

# # def AutoScroll(driver):
# #     z = 1000
# #     previous_height = -math.inf
# #     while True:
# #         z += 1000
# #         current_height = driver.execute_script("return document.documentElement.scrollHeight")
# #         if current_height == previous_height:
# #             break
# #         previous_height = current_height
# #         scroll = "window.scrollTo(0," + str(z) + ")"
# #         driver.execute_script(scroll)
# #         time.sleep(5)
# #         z += 1000


# # if __name__ == "__main__":

# #     driver = webdriver.Chrome()

   
# #     urls = ['https://www.beautylish.com/b/danessa-myricks-beauty?tag=cheeks#results',
# #             'https://www.beautylish.com/b/danessa-myricks-beauty?tag=eye-makeup#results',
# #             'https://www.beautylish.com/b/danessa-myricks-beauty?tag=face-makeup#results',
            
# #             'https://www.beautylish.com/b/danessa-myricks-beauty?tag=makeup#results',
# #             'https://www.beautylish.com/shop/browse?tag=facial-cleanser',
# #             'https://www.beautylish.com/shop/browse?tag=scrubs-exfoliants#results',
# #             'https://www.beautylish.com/shop/browse?tag=womens-fragrance',
# #             'https://www.beautylish.com/shop/browse?tag=mens-fragrance',
# #             'https://www.beautylish.com/shop/browse?tag=hair-scalp-treatments',
# #             'https://www.beautylish.com/shop/browse?tag=nail-color',
# #             'https://www.beautylish.com/shop/browse?tag=nail-care',
# #             'https://www.beautylish.com/shop/browse?tag=hand-treatments',
# #             'https://www.beautylish.com/shop/browse?tag=body-moisturizers',
# #             'https://www.beautylish.com/shop/browse?tag=shampoo',
# #             'https://www.beautylish.com/b/danessa-myricks-beauty?tag=lips#results',
# #             'https://www.beautylish.com/b/danessa-myricks-beauty?tag=makeup-palettes-sets#results',
# #             'https://www.beautylish.com/b/danessa-myricks-beauty?tag=makeup-removers#results',
# #             'https://www.beautylish.com/b/danessa-myricks-beauty?tag=makeup-tools#results',
# #             'https://www.beautylish.com/shop/browse?tag=candles-home-scents',
# #             'https://www.beautylish.com/shop/browse?tag=sunscreen',
# #             'https://www.beautylish.com/shop/browse?tag=conditioner'
# #             ]
# #     all_titles = []

# #     for url in urls:
# #         driver.get(url)
# #         AutoScroll(driver)

# #         content = driver.page_source
# #         soup = BeautifulSoup(content, "html.parser")
# #         titles = soup.find_all('p', class_='shopTile_itemName')

# #         titles_text = [title.text.strip() for title in titles]
# #         all_titles.append(titles_text)

# #         print("Titles scraped from", url, ":", titles_text)

# #     driver.quit()

# #     with open("data.csv", "w", newline='', encoding="utf-8") as csvfile:
# #         csvwriter = csv.writer(csvfile)
# #         for titles in all_titles:
# #             csvwriter.writerow(titles)

# #     print("Total titles scraped:", sum(len(titles) for titles in all_titles))

# import csv
# from bs4 import BeautifulSoup
# from selenium import webdriver
# import math
# import time

# def AutoScroll(driver):
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


# if __name__ == "__main__":
#     driver = webdriver.Chrome()

#     urls =['https://health.ucdavis.edu/blog/cultivating-health/category/exercise-and-fitness',
#            'https://health.ucdavis.edu/blog/cultivating-health/category/children-health',
#            'https://health.ucdavis.edu/blog/cultivating-health/category/exercise-and-fitness',
#            'https://health.ucdavis.edu/blog/cultivating-health/category/mental-health',
#            'https://health.ucdavis.edu/blog/cultivating-health/archive',
#            'https://health.ucdavis.edu/blog/cultivating-health/category/cancer',
#            'https://health.ucdavis.edu/blog/cultivating-health',
#         #    'https://health.ucdavis.edu/welcome/',
#         #    'https://health.ucdavis.edu/patients-visitors/',
#         #    'https://health.ucdavis.edu/research/',
#         #    'https://health.ucdavis.edu/news/',
#         #    'https://health.ucdavis.edu/healthcare-professionals/',
#         #    'https://health.ucdavis.edu/patients-visitors/services-specialties/',
#         #    'https://health.ucdavis.edu/schools-programs/',
#         #    'https://health.ucdavis.edu/about/'
#             ]

#     for index, url in enumerate(urls):
#         driver.get(url)
#         AutoScroll(driver)

#         content = driver.page_source
#         soup = BeautifulSoup(content, "html.parser")
#         titles = soup.find_all('h2', class_='card-title mt-0 mb-2 mb-sm-3')
#         titles_text = [title.text.strip() for title in titles]

#         filename = f"healthdata_{index}.csv"
#         with open(filename, "w", newline='', encoding="utf-8") as csvfile:
#             csvwriter = csv.writer(csvfile)
#             csvwriter.writerow(["Title"])
#             csvwriter.writerows(zip(titles_text))

#         print("Titles scraped from", url, "saved to", filename)

#     driver.quit()





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

    urls =['https://health.ucdavis.edu/blog/cultivating-health/category/exercise-and-fitness',
           'https://health.ucdavis.edu/blog/cultivating-health/category/children-health',
           'https://health.ucdavis.edu/blog/cultivating-health/category/exercise-and-fitness',
           'https://health.ucdavis.edu/blog/cultivating-health/category/mental-health',
           'https://health.ucdavis.edu/blog/cultivating-health/archive',
           'https://health.ucdavis.edu/blog/cultivating-health/category/cancer',
           'https://health.ucdavis.edu/blog/cultivating-health'
            ]

    for index, url in enumerate(urls):
        driver.get(url)
        AutoScroll(driver)

        content = driver.page_source
        soup = BeautifulSoup(content, "html.parser")
        titles = soup.find_all('h2', class_='card-title mt-0 mb-2 mb-sm-3')
        titles_text = [title.text.strip() for title in titles]

        filename = f"data_{index}.txt"
        with open(filename, "w", encoding="utf-8") as txtfile:
            for title in titles_text:
                txtfile.write(title + "\n")

        print("Titles scraped from", url, "saved to", filename)

    driver.quit()

#----------- from everyday and health data


# from bs4 import BeautifulSoup
# from selenium import webdriver
# import math
# import time

# def AutoScroll(driver):
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


# if __name__ == "__main__":
#     driver = webdriver.Chrome()

#     urls =['https://www.everydayhealth.com/wellness/',
#            'https://www.everydayhealth.com/lifestyle/food/',
#            'https://www.everydayhealth.com/drugs/',
#            'https://www.everydayhealth.com/emotional-health/all-articles/',
#            'https://www.everydayhealth.com/sexual-health/sexually-transmitted-diseases/',
#            'https://www.everydayhealth.com/diet-nutrition/the-dash-diet.aspx',
#            'https://www.everydayhealth.com/wellness/healthy-skin/',
#            'https://www.everydayhealth.com/self-care/']
            

#     for index, url in enumerate(urls):
#         driver.get(url)
#         AutoScroll(driver)

#         content = driver.page_source
#         soup = BeautifulSoup(content, "html.parser")
#         titles = soup.find_all('div', class_='landing-content-list__item-dek')
#         titles_text = [title.text.strip() for title in titles]

#         filename = f"data_{index}.txt"
#         with open(filename, "w", encoding="utf-8") as txtfile:
#             for title in titles_text:
#                 txtfile.write(title + "\n")

#         print("Titles scraped from", url, "saved to", filename)

#     driver.quit()

