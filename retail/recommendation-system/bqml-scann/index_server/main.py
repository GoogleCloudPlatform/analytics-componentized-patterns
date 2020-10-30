# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from flask import Flask
from flask import request
from flask import jsonify

from lookup import EmbeddingLookup
from matching import ScaNNMatcher

PROJECT_ID = os.environ['PROJECT_ID']
REGION = os.environ['REGION']
EMBEDDNIG_LOOKUP_MODEL_NAME = os.environ['EMBEDDNIG_LOOKUP_MODEL_NAME']
EMBEDDNIG_LOOKUP_MODEL_VERSION = os.environ['EMBEDDNIG_LOOKUP_MODEL_VERSION']
INDEX_DIR = os.environ['INDEX_DIR']
PORT = os.environ['PORT']


scann_matcher = ScaNNMatcher(INDEX_DIR)
embedding_lookup = EmbeddingLookup(
    PROJECT_ID, REGION, EMBEDDNIG_LOOKUP_MODEL_NAME, EMBEDDNIG_LOOKUP_MODEL_VERSION)

app = Flask(__name__)


@app.route("/v1/models/<model>/versions/<version>", methods=["GET"])
def health(model, version):
  return jsonify({})


@app.route("/v1/models/<model>/versions/<version>:predict", methods=["POST"])
def predict(model, version):
  result = 'predictions'
  try:
    data = request.get_json()['instances'][0]
    query = data.get('query', None)
    show = data.get('show', 10)
    if not str(show).isdigit(): show = 10

    is_valid, error = validate_request(query, show)

    if not is_valid: 
      value = error
    else:
      vector = embedding_lookup.lookup([query])[0]
      value = scann_matcher.match(vector, int(show))

  except Exception as error:
    value = 'Unexpected error: {}'.format(error)
    result = 'error'

  response = jsonify({result: value})
  return response


def validate_request(query, show):
  is_valid = True
  error = ''

  if not query:
    is_valid = False
    error = 'You need to provide the item Id(s) in the query!'

  return is_valid, error


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=PORT)