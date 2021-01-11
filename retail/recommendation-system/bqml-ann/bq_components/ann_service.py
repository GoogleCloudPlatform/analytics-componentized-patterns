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
"""Helper classes encapsulating ANN Service REST API."""

import datetime
import logging
import json
import time

import google.auth

class ANNClient(object):
    """Base ANN Service client."""
    
    def __init__(self, project_id, project_number, region):
        credentials, _ = google.auth.default()
        self.authed_session = google.auth.transport.requests.AuthorizedSession(credentials)
        self.ann_endpoint = f'{region}-aiplatform.googleapis.com'
        self.ann_parent = f'https://{self.ann_endpoint}/v1alpha1/projects/{project_id}/locations/{region}'
        self.project_id = project_id
        self.project_number = project_number
        self.region = region
        
    def wait_for_completion(self, operation_id, message, sleep_time):
        """Waits for a completion of a long running operation."""
        
        api_url = f'{self.ann_parent}/operations/{operation_id}'

        start_time = datetime.datetime.utcnow()
        while True:
            response = self.authed_session.get(api_url)
            if response.status_code != 200:
                raise RuntimeError(response.json())
            if 'done' in response.json().keys():
                logging.info('Operation completed!')
                break
            elapsed_time = datetime.datetime.utcnow() - start_time
            logging.info('{}. Elapsed time since start: {}.'.format(
                message, str(elapsed_time)))
            time.sleep(sleep_time)
            
            print(response)
            print(response.json())
    
        return response.json()['response']


class IndexClient(ANNClient):
    """Encapsulates a subset of control plane APIs 
    that manage ANN indexes."""

    def __init__(self, project_id, project_number, region):
        super().__init__(project_id, project_number, region)

    def create_index(self, display_name, description, metadata):
        """Creates an ANN Index."""
    
        api_url = f'{self.ann_parent}/indexes'
    
        request_body = {
            'display_name': display_name,
            'description': description,
            'metadata': metadata
        }
    
        response = self.authed_session.post(api_url, data=json.dumps(request_body))
        if response.status_code != 200:
            raise RuntimeError(response.text)
        operation_id = response.json()['name'].split('/')[-1]
        
        return operation_id

    def list_indexes(self, display_name=None):
        """Lists all indexes with a given display name or
        all indexes if the display_name is not provided."""
    
        if display_name:
            api_url = f'{self.ann_parent}/indexes?filter=display_name="{display_name}"'
        else:
            api_url = f'{self.ann_parent}/indexes'

        response = self.authed_session.get(api_url).json()

        return response['indexes'] if response else []
    
    def delete_index(self, index_id):
        """Deletes an ANN index."""
        
        api_url = f'{self.ann_parent}/indexes/{index_id}'
        response = self.authed_session.delete(api_url)
        if response.status_code != 200:
            raise RuntimeError(response.text)


class IndexDeploymentClient(ANNClient):
    """Encapsulates a subset of control plane APIs 
    that manage ANN endpoints and deployments."""
    
    def __init__(self, project_id, project_number, region):
        super().__init__(project_id, project_number, region)

    def create_endpoint(self, display_name, vpc_name):
        """Creates an ANN endpoint."""
    
        api_url = f'{self.ann_parent}/indexEndpoints'
        network_name = f'projects/{self.project_number}/global/networks/{vpc_name}'

        request_body = {
            'display_name': display_name,
            'network': network_name
        }

        response = self.authed_session.post(api_url, data=json.dumps(request_body))
        if response.status_code != 200:
            raise RuntimeError(response.text)
        operation_id = response.json()['name'].split('/')[-1]
    
        return operation_id
    
    def list_endpoints(self, display_name=None):
        """Lists all ANN endpoints with a given display name or
        all endpoints in the project if the display_name is not provided."""
        
        if display_name:
            api_url = f'{self.ann_parent}/indexEndpoints?filter=display_name="{display_name}"'
        else:
            api_url = f'{self.ann_parent}/indexEndpoints'

        response = self.authed_session.get(api_url).json()
 
        return response['indexEndpoints'] if response else []
    
    def delete_endpoint(self, endpoint_id):
        """Deletes an ANN endpoint."""
        
        api_url = f'{self.ann_parent}/indexEndpoints/{endpoint_id}'
        
        response = self.authed_session.delete(api_url)
        if response.status_code != 200:
            raise RuntimeError(response.text)
        
        return response.json()
    
    def create_deployment(self, display_name, deployment_id, endpoint_id, index_id):
        """Deploys an ANN index to an endpoint."""
    
        api_url = f'{self.ann_parent}/indexEndpoints/{endpoint_id}:deployIndex'
        index_name = f'projects/{self.project_number}/locations/{self.region}/indexes/{index_id}'

        request_body = {
            'deployed_index': {
                'id': deployment_id,
                'index': index_name,
                'display_name': display_name
            }
        }

        response = self.authed_session.post(api_url, data=json.dumps(request_body))
        if response.status_code != 200:
            raise RuntimeError(response.text)
        operation_id = response.json()['name'].split('/')[-1]
        
        return operation_id
    
    def get_deployment_grpc_ip(self, endpoint_id, deployment_id):
        """Returns a private IP address for a gRPC interface to 
        an Index deployment."""
  
        api_url = f'{self.ann_parent}/indexEndpoints/{endpoint_id}'

        response = self.authed_session.get(api_url)
        if response.status_code != 200:
            raise RuntimeError(response.text)
            
        endpoint_ip = None
        if 'deployedIndexes' in response.json().keys():
            for deployment in response.json()['deployedIndexes']:
                if deployment['id'] == deployment_id:
                    endpoint_ip = deployment['privateEndpoints']['matchGrpcAddress']
                    
        return endpoint_ip

    
    def delete_deployment(self, endpoint_id, deployment_id):
        """Undeployes an index from an endpoint."""
        
        api_url = f'{self.ann_parent}/indexEndpoints/{endpoint_id}:undeployIndex'
        
        request_body = {
            'deployed_index_id': deployment_id
        }
    
        response = self.authed_session.post(api_url, data=json.dumps(request_body))
        if response.status_code != 200:
            raise RuntimeError(response.text)
        
        return response
    
