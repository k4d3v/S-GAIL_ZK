IMAGE_NAME="sgail_zlatko"

# IMAGE_NAME="tensorflow/tensorflow:0.12.0-gpu-py3"

docker run --rm -it \
    --privileged \
    --gpus all \
    -e LOCAL_UID=$(id -u $USER) \
    -e LOCAL_GID=$(id -g $USER) \
    -v $(pwd)/../:/home/developer/S-GAIL_ZK \
    -p 8888-8899:8888 \
    -p 6006-6010:6006 \
    -p 5900-5902:5900 \
    --name $(id -u -n)-sgail_zlatko \
    ${IMAGE_NAME} bash
