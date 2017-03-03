from flask import Flask, request
import base64
import shutil
import os
from scripts.main_top_level import main
from scripts.updateMetadata import main as update_metadata
from scripts.pushToSolr import main as push
app = Flask(__name__)


def start_extraction(doi, dir_name, user):
    main(dir_name, 0, None, 0, doi)
    with open(dir_name+'output.json', 'r') as output:
        result = output.read()
    shutil.rmtree(dir_name)
    push(str(result), user)
    return str(result)


@app.route('/')
def index():
    return "Hello, World!\n"


@app.route('/extraction', methods=['POST'])
def extract():
    request.get_data()
    data = request.form
    pdf = base64.b64decode(data['file'])
    doi = data['doi']
    user = data['user']
    dir_name = doi.replace('/', '-')
    dir_name += '/'
    try:
        os.makedirs(dir_name)
    except Exception as e:
        print e
        return 'Someone is already modifying this document'
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

