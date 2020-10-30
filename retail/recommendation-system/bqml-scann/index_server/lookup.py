# Copyright 2020 Google LLC
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

import googleapiclient.discovery
from google.api_core.client_options import ClientOptions


class EmbeddingLookup(object):
    
  def __init__(self, project, region, model_name, version):
    api_endpoint = f'https://{region}-ml.googleapis.com'
    client_options = ClientOptions(api_endpoint=api_endpoint)
    self.service = googleapiclient.discovery.build(
      serviceName='ml', version='v1', client_options=client_options)
    self.name = f'projects/{project}/models/{model_name}/versions/{version}'
    print(f'Embedding lookup service {self.name} is initialized.')
    
  def lookup(self, instances):
    request_body = {'instances': instances}
    response = self.service.projects().predict(name=self.name, body=request_body).execute()

    if 'error' in response:
      raise RuntimeError(response['error'])

    return response['predictions']