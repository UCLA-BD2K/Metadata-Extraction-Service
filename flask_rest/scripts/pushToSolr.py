import json
import re
import subprocess
from json import JSONDecoder
from functools import partial
import sys
import shutil
from pysolr4 import Solr
port = 8983
to_insert = []
to_fetch = 100000000 # all documents in Solr database
id_doi = dict()


def json_parse(fileobj, decoder=JSONDecoder(), buffersize=2048):
    buf = ''
    for chunk in iter(partial(fileobj.read, buffersize), ''):
        buf += chunk
        while buf:
            try:
                result, index = decoder.raw_decode(buf)
                yield result
                buf = buf[index:]
            except ValueError:
                # Not enough data to decode, read more
                break


def set_id_owners_file(f, cur_id, owners):
    for obj in json_parse(f):
        print "Found obj with doi " + obj['publicationDOI']
        id = get_id_from_solr(obj['publicationDOI'])
        if id is None:
            id = cur_id
            increment = True
        else:
            increment = False
        if owners:
            obj["owners"] = owners
        obj["id"] = id

        to_insert.append(obj)
        if increment:
            cur_id += 1


def set_id_owners(obj, cur_id, owners):
    print "Found obj with doi " + obj['publicationDOI']
    id = get_id_from_solr(obj['publicationDOI'])
    if id is None:
        id = cur_id
        increment = True
    else:
        increment = False
    if owners:
        obj["owners"] = owners
    obj["id"] = id

    to_insert.append(obj)
    if increment:
        cur_id += 1

def push_to_solr(output):
    output = json.dumps(output)
    subprocess.call(
        [
            "curl",
            "-X",
            "-POST",
            "-H",
            "Content-Type: application/json",
            "http://localhost:" +
            str(port) +
            "/solr/BD2K/update/json/docs/?commit=true",
            "--data-binary",
            output
        ])

'''
ID system: Id's should be consecutive positive integers. Uniqueness is based off of DOI number.
'''


def get_starting_id():
    solr = Solr('http://localhost:8983/solr/BD2K')
    result = solr.select(('q', '*:*'), ('rows', str(to_fetch)), ('wt', 'json'),
                         ('fl', 'id, publicationDOI'))
    for doc in result.docs:
        if 'id' in doc and 'publicationDOI' in doc:
            id_doi[doc['publicationDOI']] = doc['id']

    return len(result.docs)+1


def get_id_from_solr(doi):
    if doi in id_doi:
        return id_doi[doi]
    return None


def main(data, owners=None):
    print "Data in pushToSolr is "
    print data
    cur_id = get_starting_id()
    try:
        with open(data, 'rU') as f:
            set_id_owners_file(f, cur_id, owners)
    except:
        # Data must be json blob if can't open as file
            print "Setting id owners with start id = " + str(cur_id)
            set_id_owners(json.loads(data), cur_id, owners)
            #shutil.rmtree('test.json')

    print "Total size is " + str(len(to_insert))
    print to_insert
    for obj in to_insert:
        push_to_solr(obj)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])