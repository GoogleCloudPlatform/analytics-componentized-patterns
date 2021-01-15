## Setting up the ANN Service Experimental release

This document outlines the steps required to enable and configure the Experimental release of the AI Platform ANN service. 
This instructions will be updated when the service moves to the Preview and General Availability.

### Allow-listing the project

Contact your Google representative to allow-list your project and user id(s).

### Enabling the Cloud APIs required by the ANN Service

You need to enable the following APIs to use the ANN service:

* aiplatform.googleapis.com
* servicenetworking.googleapis.com
* compute.googleapis.com

### Configuring private IP access to the ANN Service

In the experimental release, ANN service is only accessible using private endpoints. Before using the service you need to have a [VPC network](https://cloud.google.com/vpc) configured with [private services access](https://cloud.google.com/vpc/docs/configure-private-services-access). You can use the `default` VPC or create a new one.

The below instructions are for a VPC that was created with auto subnets and regional dynamic routing mode (defaults). It is recommended that you execute the below commands from Cloud Shell, using the account with the `roles/compute.networkAdmin` permissions.

1. Set environment variables for your project ID, the name of your VPC network, and the name of your reserved range of addresses. The name of the reserved range can be an arbitrary name. It is for display only.

```
PROJECT_ID=<your-project-id>
gcloud config set project $PROJECT_ID
NETWORK_NAME=<your-VPC-network-name>
PEERING_RANGE_NAME=google-reserved-range

```

2. Reserve an IP range for Google services. The reserved range should be large enought to accommodate all peered services. The below command reserves a CIDR block with mask /16

```
gcloud compute addresses create $PEERING_RANGE_NAME \
  --global \
  --prefix-length=16 \
  --description="peering range for Google service: AI Platform Online Prediction" \
  --network=$NETWORK_NAME \
  --purpose=VPC_PEERING \
  --project=$PROJECT_ID

```

3. Create a private connection to establish a VPC Network Peering between your VPC network and the Google services network.

```
gcloud services vpc-peerings connect \
  --service=servicenetworking.googleapis.com \
  --network=$NETWORK_NAME \
  --ranges=$PEERING_RANGE_NAME \
  --project=$PROJECT_ID

```

