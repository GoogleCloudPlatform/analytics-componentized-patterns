export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export IMAGE_REPO_NAME=hotel_recommender_vertexai_container
export IMAGE_TAG=latest
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

docker build --no-cache -f Dockerfile -t $IMAGE_URI ./
docker push $IMAGE_URI