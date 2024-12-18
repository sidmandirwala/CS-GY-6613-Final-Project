import os
import shutil
import subprocess
import tempfile
import uuid
from typing import List, Dict
from loguru import logger
from pymongo import MongoClient, errors

MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "github_scraper"
COLLECTION_NAME = "repositories"

client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

class GithubCrawler:
    def __init__(self, ignore=(".git", ".toml", ".lock", ".png")) -> None:
        self._ignore = ignore

    def extract(self, link: str, user: Dict) -> None:
        """Extracts content from a GitHub repository and saves it to MongoDB."""
        # Check if repository already exists
        if collection.find_one({"link": link}):
            logger.info(f"Repository already exists in the database: {link}")
            return

        logger.info(f"Starting to scrape GitHub repository: {link}")
        repo_name = link.rstrip("/").split("/")[-1]
        local_temp = tempfile.mkdtemp()

        try:
            # Clone the repository
            subprocess.run(["git", "clone", link], check=True, cwd=local_temp)

            repo_path = os.path.join(local_temp, os.listdir(local_temp)[0])

            tree = {}
            for root, _, files in os.walk(repo_path):
                rel_dir = os.path.relpath(root, repo_path)
                if any(rel_dir.startswith(pattern) for pattern in self._ignore):
                    continue

                for file in files:
                    if any(file.endswith(pattern) for pattern in self._ignore):
                        continue
                    file_path = os.path.join(rel_dir, file) if rel_dir != '.' else file
                    try:
                        with open(os.path.join(root, file), "r", errors="ignore") as f:
                            tree[file_path] = f.read().strip()
                    except Exception as e:
                        logger.warning(f"Failed to read file {file_path}: {e}")

            repo_data = {
                "_id": str(uuid.uuid4()),
                "name": repo_name,
                "link": link,
                "content": tree,
                "platform": "github",
                "author_id": user["id"],
                "author_full_name": user["full_name"],
            }
            collection.insert_one(repo_data)
            logger.info(f"Repository {repo_name} saved successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e}")
        except errors.PyMongoError as e:
            logger.error(f"Failed to save data to MongoDB: {e}")
        finally:
            shutil.rmtree(local_temp)

        logger.info(f"Finished scraping GitHub repository: {link}")

    def process_links(self, links: List[str], user: Dict) -> None:
        """Processes a list of GitHub repository links."""
        for link in links:
            self.extract(link, user)

if __name__ == "__main__":
    crawler = GithubCrawler()
    test_user = {"id": str(uuid.uuid4()), "full_name": "Test User"}
    
    github_links = [
        "https://github.com/ros-controls/ros2_controllers",
        "https://github.com/ros2/ros2"
    ]
    
    crawler.process_links(links=github_links, user=test_user)
