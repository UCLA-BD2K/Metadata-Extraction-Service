# Metadata-Extraction-Service
Python scripts including Flask REST app for metadata extraction

PYTHON (2.7) SCRIPTS FOR METADATA EXTRACTION

1. checkForPublications.py --> Queries pubmed with n days as a parameter to find new publications which have come out in the past n days. Targets specific journals, list can be expanded using journals.txt.

2. downloadPublications.py --> Downloads publications as pdfs to a folder, takes as input a txt file containing doi numbers or pmids.

	Usage: python downloadPublications.py file.txt directoryToSaveFilesIn/ (optional)

3. main_top_level.py --> All-in-one script that extracts metadata from PDF using GROBID, enriches using APIs, and inserts into Solr. Each component is modularized (1. Get papers from Journal (Download PDFs if needed) 2. Classify publication 3. Extract metadata from PDF & enrich 4. Insert metadata into Solr)

	Usage: python main_top_level.py -directory pdfDir/ -pushToSolr 1/0 -doiRecords dois.json

	dois.json is obtained automatically from using downloadPublications.py
	
	Calls the following scripts in order:
	classifier.py (Still in progress, current accuracy ~80%)
	pdf_extract.py
	parse_extracts.py
	pushToSolr.py

	Dependencies ==>

	Solr instance running at localhost:8983 (if pushing to Solr)

	GROBID (https://github.com/kermitt2/grobid) running on localhost:8080

	Tor and Privoxy for downloading/web scraping (Follow instructions on these pages for Linux and OS X respectively:
	http://sacharya.com/crawling-anonymously-with-tor-in-python/ and http://www.andrewwatters.com/privoxy/)
	Tor control port is 9051, set password as 'downloader' in torrc file. If you want to set a different password, please change the python file.
	Privoxy running on port 8118

	pdftotext

	sudo pip install bs4
	git clone git://github.com/aaronsw/pytorctl.git
	pip install pytorctl/
	sudo pip install pycurl
	sudo pip install xmltodict
	sudo pip install summa
	sudo pip install -U nltk (download nltk.tokenize)

	Scipy
	NumPy
	Scikit 0.18 dev version

4. schema.xml and solrconfig.xml --> Configure your local solr server with these files to run the above scripts.

