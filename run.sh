#/bin/bash
DOCKER_BUILDKIT=1 docker build -t mpi-satellite .

docker run --name mpi-satellite \
    --rm \
    -e "HOST=0.0.0.0" \
    -e "PORT=8000" \
    -e "sh_client_id=35c2e8f8-8796-450a-9770-f4e481949986" \
    -e "sh_client_secret=Y*X?HpB9W4N&#lWKt!1CQL3cj-)r(6M{n}Nuj6QD"  \
    -p "8000:8000" \
    mpi-satellite