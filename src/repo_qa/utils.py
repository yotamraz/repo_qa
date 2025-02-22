import time

import requests
import uvicorn
import git
from loguru import logger

from .api import app

def wait_for_server(url, timeout=10):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                logger.info("API server is up!")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    logger.info("API server did not start in time.")
    return False

def run_api_server(host: str, port: int):
    uvicorn.run(app, host=host, port=port)

def get_git_diff(repo_path, rev_range="HEAD~1..HEAD"):
    """
    Returns the Git diff as a string for the given rev_range,
    e.g. 'HEAD~1..HEAD' means changes between the previous commit and HEAD.
    """
    repo = git.Repo(repo_path)
    diff_text = repo.git.diff(rev_range)
    return diff_text