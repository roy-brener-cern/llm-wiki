import requests
from bs4 import BeautifulSoup
import http.cookiejar as cookielib
import os
import re
import time

# -------------------------
# CONFIG
# -------------------------

# Root page to start fetching
root_url = "https://twiki.cern.ch/twiki/bin/view/AtlasProtected/EgammaCalibration?skin=print"

# Path to your exported cookies from Cookie-Editor
cookie_file = "twiki_cookies.txt"

# Folder to save pages
save_folder = "pages"
os.makedirs(save_folder, exist_ok=True)

# -------------------------
# COOKIE + SESSION
# -------------------------
cj = cookielib.MozillaCookieJar(cookie_file)
cj.load(ignore_discard=True, ignore_expires=True)

session = requests.Session()
session.cookies = cj

# -------------------------
# UTILITIES
# -------------------------

def url_to_filename(url):
    """Convert a Twiki URL to a safe local filename"""
    name = url.replace("https://twiki.cern.ch/twiki/bin/view/AtlasProtected/", "")
    name = re.sub(r"[^0-9a-zA-Z_-]", "_", name)
    return os.path.join(save_folder, f"{name}.html")

def fetch_and_save(session, url, visited=None):
    """Fetch a page, save it if missing, and recursively fetch subpages."""
    if visited is None:
        visited = set()

    if url in visited:
        return
    visited.add(url)

    filename = url_to_filename(url)

    if os.path.exists(filename):
        print(f"Skipping {filename}, already downloaded")
    else:
        try:
            resp = session.get(url)
            if resp.status_code != 200:
                print(f"Failed {url} with status {resp.status_code}")
                return
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return

        with open(filename, "w", encoding="utf-8") as f:
            f.write(resp.text)
        print(f"Saved {url} -> {filename}")

    # Parse page for subpage links
    soup = BeautifulSoup(open(filename, "r", encoding="utf-8"), "html.parser")
    sub_links = [
        "https://twiki.cern.ch" + a["href"]
        for a in soup.find_all("a", href=True)
        if a["href"].startswith("/twiki/bin/view/AtlasProtected/") and
           "https://twiki.cern.ch" + a["href"] not in visited
    ]

    for link in sub_links:
        time.sleep(0.5)  # polite delay
        fetch_and_save(session, link, visited)

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    fetch_and_save(session, root_url)
