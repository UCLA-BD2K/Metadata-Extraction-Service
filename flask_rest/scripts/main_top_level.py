import sys
import os
import json
from classifier import main as classify
from pdf_extract import main as grobid_extraction
from parse_extracts import main as parse_extracts
from pushToSolr import main as pushToSolr
import time


def main(directory, solr, doiRecords, to_classify, doi):
    start_time = time.time()

    if doi is None and doiRecords is None:
        print "Please either pass in doiRecords or single doi"
        sys.exit(1)

    directory = directory + '/' if directory[-1] is not '/' else directory
    if not os.path.isdir(directory):
        print directory + " is not a directory, creating it and continuing..."
        os.makedirs(directory)

    tools_dir = directory + 'tools/'
    non_tools_dir = directory + 'non_tools/'
    tools_xml_dir = directory + 'tools_xml/'
    tools_txt_dir = directory + 'tools_txt/'
    output_file = directory + 'output.json'

    # Classify
    if to_classify != 0:
        classify(directory, tools_dir, non_tools_dir)
    else:
        tools_dir = directory

    # Grobid extraction
    grobid_extraction(tools_dir, tools_xml_dir, tools_txt_dir)

    # Parse grobid data into output file
    parse_extracts(tools_xml_dir, tools_txt_dir, doiRecords, output_file, doi)

    # Push to Solr if needed
    if solr is not None and solr == 1:
        pushToSolr(output_file)

    print(" Total time taken:    --- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]))
