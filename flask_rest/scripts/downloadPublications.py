#!/usr/bin/python
import urllib
from bs4 import BeautifulSoup
import sys
import json
import re
import xml.etree.ElementTree as ET
import urllib2
import Queue
from threading import Thread
import random
import time
import pycurl
from TorCtl import TorCtl
import os
# We should ignore SIGPIPE when using pycurl.NOSIGNAL - see
# the libcurl tutorial for more info.
try:
    import signal
    from signal import SIGPIPE, SIG_IGN
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
except ImportError:
    pass

# Dummy user agents for authenticity
user_agent = [
    'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7',
    'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) Gecko/20100101 Firefox/40.1',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/7046A194A']

pmids = []
oxford_ext = ".full.pdf"
nature_ext = ".pdf"
doiData = dict()
docs = []
num_threads = 4  # Threads to use for crossref and pubmed API's
extract_failed = False  # Set to true when extraction failed due to bandwidth limit exceeded
password = "downloader"  # Tor config password
tor_control_port = 9051


def _set_urlproxy():
    '''
    Configures urllib2 to work with privoxy running on localhost port 8118
    '''
    proxy_support = urllib2.ProxyHandler({"http": "127.0.0.1:8118"})
    opener = urllib2.build_opener(proxy_support)
    urllib2.install_opener(opener)


def get_ip_address():
    url = "http://icanhazip.com/"
    _set_urlproxy()
    headers = {'User-Agent': user_agent[random.randint(0, 3)]}
    request = urllib2.Request(url, None, headers)
    return urllib2.urlopen(request).read()


def renew_connection():
    '''
    When bandwidth limit is exceeded (http error 509), this function is
    called to change ip address by establishing new tor connection
    '''
    old_ip = get_ip_address()
    print "Old ip address is " + old_ip
    conn = TorCtl.connect(
        controlAddr="127.0.0.1",
        controlPort=tor_control_port,
        passphrase=password)
    conn.send_signal("NEWNYM")
    conn.close()
    new_ip = get_ip_address()
    while new_ip == old_ip:
        new_ip = get_ip_address()
    print "New ip address is " + new_ip


class Document(object):
    '''
    Simple document class to store doi, name and url of document
    '''
    doi = None
    url = None
    name = None


class PmidConverter(Thread):
    '''
    Thread class which converts pmid to dois
    '''

    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            # Get the work from the queue
            pmid = self.queue.get()
            doi = pmid_doi(pmid)
            d = Document()
            d.doi = doi
            docs.append(d)
            self.queue.task_done()


class NameFetcher(Thread):
    '''
    Thread class to fetch name from doi using crossref's API
    '''

    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            # Get the work from the queue
            doc = self.queue.get()
            doc.name = get_name(doc.doi)
            self.queue.task_done()


def get_name(doi):
    '''
    Fetches name from crossref's API, removes non space and non words
    :param doi:
    :return: First 6 words of name as a string, entire name can be too long
    '''
    api_url = "http://api.crossref.org/works/" + str(doi)
    try:
        print api_url
        response = urllib.urlopen(str(api_url)).read()
        parsed_json = json.loads(str(response))
        name = parsed_json['message']['title'][0]
        # replace non words and non space with nothing to avoid issues
        name = re.sub(r'[^ \w]+', '', name)
        name_list = name.split()
        # Get first 6 words as name, if 6 less than number of words in title
        if len(name_list) < 6:
            name = ' '.join(name_list[:len(name_list)])
        else:
            name = ' '.join(name_list[:6])
        return name
    except Exception as e:
        print e
        return "Name not found, crossref name error"


def pmid_doi(pmid):
    '''
    Convert pmid to doi using pubmed's API
    :param pmid:
    :return: doi number
    '''
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&rettype=xml&id=" + \
        str(pmid)
    try:
        print "nih converter url is " + url
        response = urllib.urlopen(str(url)).read()
        root = ET.fromstring(response)
        return root[0][1][2][1].text
    except Exception as e:
        print "DOI not found in eutils"
        print e
        return None


def get_page(url):
    '''
    This function gets the page url by following the url provided from the doi by crossref's API
    Function is necessary as crossref's API url is sometimes redirected, this returns final url of
    publication page
    :param url:
    :return: publication url
    '''
    if "nature" in url:
        # Nature has to be handled separately, so return
        return url

    _set_urlproxy()
    headers = {'User-Agent': user_agent[random.randint(0, 3)]}
    global extract_failed
    try:
        req = urllib2.Request(url, None, headers)
        res = urllib2.urlopen(req)
        extract_failed = False
        return res.geturl()
    except urllib2.HTTPError as e:
        print e
        print "Download url fetching failed from " + url
        if e.code == 509:
            print "Renewing connection and trying again"
            extract_failed = True
        return None
    except Exception as e:
        print e
        print "Download url fetching failed from " + url
        return None


def handle_nature(doi):
    '''
    For nature, the url has to be constructed manually using crossref's api to find
    out the volume and issue number
    :param doi:
    :return: pdf url of publication, to be downloaded
    '''
    try:
        url = "http://api.crossref.org/works/" + doi
        response = urllib2.urlopen(url).read()
        parsed_json = json.loads(str(response))
        issue = parsed_json['message']['issue']
        volume = parsed_json['message']['volume']
    except Exception as e:
        print e
        print "Crossref nature extraction failed"
        return None
    suffix = str(doi).split("/")[1]
    return "http://nature.com/nbt/journal/v"+volume+"/n"+issue+"/pdf/"+suffix+".pdf"


def get_final_url(final_url, doi):
    '''
    Called after get_page, this function constructs final pdf url based on journal
    :param final_url:
    :param doi:
    :return: final pdf url, to be downloaded
    '''
    if "oxford" in final_url:
        return final_url + oxford_ext
    if "nature" in final_url:
        return handle_nature(doi)
    if "bmcbioinformatics" in final_url:
        return "http://bmcbioinformatics.biomedcentral.com/track/pdf/"+doi+"?site=bmcbioinformatics.biomedcentral.com"


def extract_url(doc):
    '''
    Returns final url to be downloaded using doi by setting doc.url field
    :param doc:
    :return:
    '''
    try:
        url = "http://dx.doi.org/api/handles/" + str(doc.doi)
        response = urllib.urlopen(str(url)).read()
        parsed_json = json.loads(str(response))
        lookup_url = parsed_json['values'][0]['data']['value']
        print "Calling get_page with url " + lookup_url
        final_url = get_page(lookup_url)
        if final_url is None:
            return
        doc.url = get_final_url(final_url, doc.doi)
        if doc.url is None:
            return
        print "Manually extracted url is " + doc.url
    except Exception as e:
        print e
        print "Manual extraction of url failed"
        if doc.doi is not None:
            print "Failed with doi number " + doc.doi
        doc.url = None


def save_doi_data():
    '''
    Save publication data as key value name:doi pairs, used later by other scripts.
    :return:
    '''
    for doc in docs:
        doiData[doc.name] = doc.doi
    with open('dois.json', 'w') as outfile:
        outfile.write(json.dumps(doiData, indent=2))


def start_conversion():
    '''
    Start PMID to DOI conversion
    :return:
    '''
    queue = Queue.Queue()
    for x in range(num_threads):
        worker = PmidConverter(queue)
        worker.daemon = True
        worker.start()

    for pmid in pmids:
        queue.put(pmid)

    queue.join()


def update_name(directory):
    '''
    Fetch names for all documents
    :param directory:
    :return:
    '''
    queue = Queue.Queue()
    for x in range(num_threads):
        worker = NameFetcher(queue)
        worker.daemon = True
        worker.start()

    for doc in docs:
        queue.put(doc)

    queue.join()

    downloaded = set(os.listdir(directory))
    global docs
    docs = [doc for doc in docs if doc.name + ".pdf" not in downloaded]


def download(directory, doc):
    '''
    Main download function: Uses pycurl for reliable downloading.
    Only one connection is used (num_conn) to avoid hammering the servers too much.
    :param directory:
    :param doc:
    :return:
    '''
    num_conn = 1
    queue = []
    url = doc.url
    filename = directory + doc.name + ".pdf"
    queue.append((url, filename))

    # Check args
    assert queue, "no URLs given"
    num_urls = len(queue)
    num_conn = min(num_conn, num_urls)
    assert 1 <= num_conn <= 10000, "invalid number of concurrent connections"
    print "PycURL %s (compiled against 0x%x)" % (pycurl.version, pycurl.COMPILE_LIBCURL_VERSION_NUM)
    print "----- Getting", num_urls, "URLs using", num_conn, "connections -----"

    # Pre-allocate a list of curl objects
    m = pycurl.CurlMulti()
    m.handles = []
    for i in range(num_conn):
        c = pycurl.Curl()
        c.fp = None
        c.setopt(pycurl.FOLLOWLOCATION, 1)
        c.setopt(pycurl.MAXREDIRS, 5)
        c.setopt(pycurl.CONNECTTIMEOUT, 30)
        c.setopt(pycurl.TIMEOUT, 300)
        c.setopt(pycurl.NOSIGNAL, 1)
        c.setopt(pycurl.USERAGENT, user_agent[random.randint(0, 3)])
        m.handles.append(c)

    # Main loop
    freelist = m.handles[:]
    num_processed = 0
    while num_processed < num_urls:
        # If there is an url to process and a free curl object, add to multi
        # stack
        while queue and freelist:
            url, filename = queue.pop(0)
            c = freelist.pop()
            c.fp = open(filename, "wb")
            c.setopt(pycurl.URL, url)
            c.setopt(pycurl.WRITEDATA, c.fp)
            m.add_handle(c)
            # store some info
            c.filename = filename
            c.url = url
        # Run the internal curl state machine for the multi stack
        while True:
            ret, num_handles = m.perform()
            if ret != pycurl.E_CALL_MULTI_PERFORM:
                break
        # Check for curl objects which have terminated, and add them to the
        # freelist
        global extract_failed
        while True:
            num_q, ok_list, err_list = m.info_read()
            for c in ok_list:
                c.fp.close()
                c.fp = None
                m.remove_handle(c)
                print "Success:", c.filename, c.url, c.getinfo(pycurl.EFFECTIVE_URL)
                freelist.append(c)
                extract_failed = False
            for c, errno, errmsg in err_list:
                c.fp.close()
                c.fp = None
                m.remove_handle(c)
                print "Failed: ", c.filename, c.url, errno, errmsg
                freelist.append(c)
                if c.getinfo(pycurl.HTTP_CODE) == 509:
                    print "Download unsuccessful for " + c.url
                    print "Limit exceeded on current IP, renewing connection and trying again"
                    extract_failed = True
            num_processed = num_processed + len(ok_list) + len(err_list)
            if num_q == 0:
                break
        # Currently no more I/O is pending, could do something in the meantime
        # (display a progress bar, etc.).
        # We just call select() to sleep until some more data is available.
        m.select(1.0)

    # Cleanup
    for c in m.handles:
        if c.fp is not None:
            c.fp.close()
            c.fp = None
        c.close()
    m.close()


def pmid_or_doi(line):
    '''
    Determine whether string contains pmid or doi
    :param line:
    :return:
    '''
    if line.isdigit():
        # We have PMID
        pmids.append(line)
    else:
        # We have DOI
        d = Document()
        d.doi = line
        docs.append(d)


def main(filename, directory=None):
    start_time = time.time()
    # Check filename and directory, create directory if needed

    if not filename:
        print "Please input file containing DOI and/or PMID numbers"
        sys.exit(1)
    if directory is None:
        directory = 'pdfDir/'
        if not os.path.isdir(directory):
            os.makedirs(directory)
    else:
        if not os.path.isdir(directory):
            os.makedirs(directory)
        directory = directory + '/' if directory[-1] is not '/' else directory

    # Process file while removing unnecessary characters

    try:
        with open(filename, 'rU') as numbers:
            for line in numbers:
                line = line.strip()
                line = line.replace('"', '')
                line = line.replace("'", '')
                line = line.replace(",", '')
                pmid_or_doi(line)
    except Exception as e:
        print e
        print "File could not be opened/processed, please check filename/path"

    if pmids:
        start_conversion()
    global docs
    # Only consider docs which have dois
    docs = [doc for doc in docs if doc.doi is not None]

    update_name(directory)

    save_doi_data()

    # Change ip before starting download
    renew_connection()

    # Extract and download, try again if extract_failed is set to true
    for doc in docs:
        extract_url(doc)
        if extract_failed:
            renew_connection()
            extract_url(doc)
        if doc.url is not None:
            download(directory, doc)
            if extract_failed:
                renew_connection()
                download(directory, doc)

    print(" Downloading time taken:    --- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    '''
    Temporary while download script is independent of pipeline
    '''
    if len(sys.argv) == 3:
        sys.exit(main(sys.argv[1], sys.argv[2]))
    elif len(sys.argv) == 2:
        sys.exit(main(sys.argv[1]))
