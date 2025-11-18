#!/usr/bin/env python3

import requests
from bs4 import BeautifulSoup
import http.cookiejar as cookielib
import os
import re
import time
from tqdm import tqdm

# -------------------------
# CONFIG
# -------------------------

# Path to your exported cookies (Netscape) from Cookie-Editor
cookie_file = "twiki_cookies.txt"

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


def make_soup(url):
    try:
        response = session.get(url)
    except:
        print( f"Failed to fetch: {url}" )
        response = None
    if response is not None:
        data = response.text
        soup = BeautifulSoup(data, "lxml")
        return soup
    else:
        return None


def store_soup(soup, filename):
    with open(filename, "w") as f:
        f.write(soup.text)


def write_to_file(links, filename):
    with open(filename, "a") as f:
        f.writelines(links)


def get_links(url, filename):
    soup = make_soup(url)
    links = []
    if soup is None:
        return []
    for link in soup.find_all("a"):
        link_url = link.get("href")
        url_found = link_url
        if link_url is not None:
            if link_url.startswith("http"):
                pass
            elif link_url.startswith("/"):
                url_found = "https://twiki.cern.ch" + link_url
                links.append(url_found + "\n")
    write_to_file(links, filename)
    return links


# -------------------------
# MAIN
# -------------------------


def main_download_links(linklistfile, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(linklistfile, "r") as f:
        for line in tqdm(f):
            #ofilename = f"{output_dir}/twiki_cern__" + line.split("AtlasProtected/")[
            ofilename = f"{output_dir}/twiki_cern__" + line.split("Atlas/")[
                -1
            ].replace("\n", "")
            if os.path.exists(ofilename):
                continue
            #if "AtlasProtected/" not in line:
            if "Atlas/" not in line:
                continue
            if "rdiff" in line:
                continue
            s = make_soup(line)
            if s is not None:
                store_soup(s, ofilename)
            else:
                print( f"No output for: {ofilename}" )


def main_get_links(letter, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    #url = f"https://twiki.cern.ch/twiki/bin/search/AtlasProtected/?scope=topic&regex=on&search=^{letter}"
    url = f"https://twiki.cern.ch/twiki/bin/search/Atlas/?scope=topic&regex=on&search=^{letter}"
    get_links(url, f"{output_dir}/{letter}.txt")


if __name__ == "__main__":
    import multiprocessing as mp
    import string
    import functools

    output_dir = "twiki_atlas"
    #output_dir = "twiki_atlas_protected"
    checks = [_ for _ in string.ascii_lowercase] + [str(_) for _ in range(10)]
    with mp.Pool(processes=min(mp.cpu_count() - 1, len(checks))) as pool:
        inputs_download = [f"{output_dir}/{check}.txt" for check in checks]
        inputs_getlinks = checks
        _main_get_links = functools.partial( main_get_links, output_dir = output_dir )
        _main_download_links = functools.partial( main_download_links, output_dir =  output_dir + "_data" )
        pool.map( _main_get_links, inputs_getlinks )
        pool.map( _main_download_links, inputs_download )
