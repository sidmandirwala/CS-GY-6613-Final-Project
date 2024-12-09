import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from bs4 import BeautifulSoup
import time
import logging
import uuid
from pymongo import MongoClient, errors
import random
from typing import List, Dict

MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "medium_scraper"
COLLECTION_NAME = "articles"

client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def random_delay(min_seconds=2, max_seconds=5):
    delay = random.uniform(min_seconds, max_seconds)
    time.sleep(delay)

class MediumCrawler:
    def __init__(self):
        self.options = uc.ChromeOptions()
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("--disable-blink-features=AutomationControlled")
        self.options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        )
        self.driver = uc.Chrome(options=self.options)

    def scroll_page(self, scroll_pause_time=1):
        """Scroll down the page to load more content."""
        last_height = self.driver.execute_script("return document.body.scrollHeight")

        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            random_delay(scroll_pause_time, scroll_pause_time + 1)

            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    def accept_cookies(self):
        """Accept cookies if the prompt appears."""
        try:
            accept_button = self.driver.find_element(By.XPATH, '//button[text()="Accept"]')
            accept_button.click()
            random_delay(2, 4)
            logger.info("Accepted cookies.")
        except NoSuchElementException:
            logger.info("No cookies prompt found.")

    def extract(self, link: str, user: Dict = None):
        """Extracts content from a Medium article and saves it to MongoDB."""
        if collection.find_one({"link": link}):
            logger.info(f"Article already exists in the database: {link}")
            return

        logger.info(f"Starting to scrape Medium article: {link}")

        self.driver.get(link)
        random_delay(3, 6)  
        self.accept_cookies()  
        self.scroll_page()

        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        
        title = soup.find("h1")
        subtitle = soup.find("h2")
        
        content_paragraphs = soup.find_all("p")
        content = "\n".join([para.get_text(strip=True) for para in content_paragraphs])
        
        data = {
            "Title": title.get_text(strip=True) if title else None,
            "Subtitle": subtitle.get_text(strip=True) if subtitle else None,
            "Content": content,
        }

        doc = {
            "_id": str(uuid.uuid4()),
            "link": link,
            "platform": "medium",
            "content": data,
        }

        if user:
            doc["author_id"] = user["id"]
            doc["author_full_name"] = user["full_name"]

        try:
            collection.insert_one(doc)
            logger.info(f"Successfully scraped and saved article: {link}")
        except errors.PyMongoError as e:
            logger.error(f"Failed to save article to MongoDB: {e}")

        return data

    def extract_multiple(self, links: List[str], user: Dict = None):
        """Extracts content from multiple Medium articles."""
        for link in links:
            try:
                self.extract(link, user=user)
            except Exception as e:
                logger.error(f"An error occurred while scraping {link}: {e}")

    def close(self):
        """Close the driver after scraping is done."""
        self.driver.quit()

if __name__ == "__main__":
    crawler = MediumCrawler()
    test_user = {"id": str(uuid.uuid4()), "full_name": "Test User"}
    
    medium_links = [
        "https://medium.com/schmiedeone/getting-started-with-ros2-part-1-d4c3b7335c71",
        "https://medium.com/@tetraengnrng/a-beginners-guide-to-ros2-29721dcf49c8"
    ]
    
    try:
        crawler.extract_multiple(links=medium_links, user=test_user)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        crawler.close()
