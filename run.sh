#/bin/bash
DOCKER_BUILDKIT=1 docker build -t mpi-satellite .

source .env

# Check that SENTINEL_CLIENT_ID and SENTINEL_CLIENT_SECRET are set
if [ -z "$SENTINEL_CLIENT_ID" ] || [ -z "$SENTINEL_CLIENT_SECRET" ]; then
  echo "Please set the SENTINEL_CLIENT_ID and SENTINEL_CLIENT_SECRET environment variables."
  exit 1
fi


docker run --name mpi-satellite \
    --rm \
    -e "HOST=0.0.0.0" \
    -e "PORT=8000" \
    -e "sh_client_id=${SENTINEL_CLIENT_ID}" \
    -e "sh_client_secret=${SENTINEL_CLIENT_SECRET}" \
    -p "8000:8000" \
    mpi-satellite