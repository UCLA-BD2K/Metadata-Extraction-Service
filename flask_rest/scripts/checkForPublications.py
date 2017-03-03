# This script queries specific journals and returns a list of DOI numbers
import sys
import urllib2
import xml.etree.ElementTree as ET
from downloadPublications import pmid_doi
# Journals supported --> Nature, Bioinformatics (Oxford) and BMC Bioinformatics, can add more from journals.txt
journal_dict = {'nature': 0410462, 'bioinformatics_oxford': 9808944, 'bmc_bioinformatics': 100965194}
dois = []
max_results = "100000"


def query_pubmed(num_days):
    for k, v in journal_dict.items():
        journal_id = str(v)
        try:
            url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term="+journal_id + \
                  "[journal]&reldate="+num_days+"&datetype=edat&retmax="+max_results
            response = urllib2.urlopen(url).read()
            root = ET.fromstring(response)
            num_results = int(root[1].text)
            if num_results <= 0:
                continue
            for i in range(num_results):
                pmid = root[3][i].text
                doi = pmid_doi(pmid)
                dois.append(doi)
        except Exception as e:
            print e
            continue
    for doi in dois:
        # Do something with list
        print doi


def main(num_days):
    if not num_days or not str(num_days).isdigit() or num_days <= 0:
        num_days = "7"
    else:
        num_days = str(num_days)
    query_pubmed(num_days)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        sys.exit(main(sys.argv[1]))
    else:
        sys.exit(main(-1))
