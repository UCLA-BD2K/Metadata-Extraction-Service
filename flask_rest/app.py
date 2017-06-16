from flask import Flask, request
import base64
import shutil
import os
from scripts.main_top_level import main
from scripts.updateMetadata import main as update_metadata
from scripts.pushToSolr import main as push
from flask import jsonify
import requests
import json
from difflib import SequenceMatcher
app = Flask(__name__)


class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


def start_extraction(doi, dir_name, user):
    print "dir name is " + dir_name
    main(dir_name, 0, None, 0, doi)
    with open(dir_name+'output.json', 'r') as output:
        result = output.read()
    shutil.rmtree(dir_name)
    if "name" not in result or SequenceMatcher(None, json.loads(result)["name"], get_title(doi)).ratio() < 0.7:
        raise InvalidUsage('The DOI you entered did not match the pdf you uploaded', status_code=406)
    push(str(result), user)
    return str(result)


def verify_doi(doi):
    crossref_url = "https://api.crossref.org/works/" + doi
    try:
        return requests.get(crossref_url).json()["status"] == "ok"
    except:
        return False


def get_title(doi):
    crossref_url = "https://api.crossref.org/works/" + doi
    return requests.get(crossref_url).json()["message"]["title"][0]


@app.route('/')
def index():
    return "Server is up and running!"


@app.route('/extraction', methods=['POST'])
def extract():
    request.get_data()
    data = request.form
    pdf = base64.b64decode(data['file'])
    doi = data['doi']
    user = data['user']
    dir_name = doi.replace('/', '-')
    dir_name += '/'
    os.makedirs(dir_name)
    #except Exception as e:
    #    print e
    #    return 'Someone is already modifying this document'
    with open(dir_name + 'file.pdf', 'wb') as input_file:
        input_file.write(pdf)
    return start_extraction(doi, dir_name, user)


@app.route('/update', methods=['POST'])
def update():
    request.get_data()
    data = request.form['data']
    return update_metadata(data)


if __name__ == '__main__':
    app.run(debug=True, port=7777)

