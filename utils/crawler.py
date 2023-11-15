import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os


def crawl(url):
    local_domain = urlparse(url).netloc

    # Create a directory to store the text files
    if not os.path.exists("text/"):
        os.mkdir("text/")

    if not os.path.exists("text/" + local_domain + "/"):
        os.mkdir("text/" + local_domain + "/")

    # Create a directory to store the csv files
    if not os.path.exists("processed"):
        os.mkdir("processed")

    with open(
        "text/" + local_domain + "/" + url[8:].replace("/", "_") + ".txt",
        "w",
        encoding="UTF-8",
    ) as f:
        soup = BeautifulSoup(requests.get(url).text, "html.parser")

        text = soup.get_text()

        # If the crawler gets to a page that requires JavaScript, it will stop the crawl
        if "You need to enable JavaScript to run this app." in text:
            print("Unable to parse page " + url + " due to JavaScript being required")

        f.write(text)
