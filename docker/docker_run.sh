IMAGE_NAME="sgail_zlatko"

docker run --rm -it \
    --privileged \
    --net=host\
    --gpus all \
    -e DISPLAY=$DISPLAY\
    -e LOCAL_UID=$(id -u $USER) \
    -e LOCAL_GID=$(id -g $USER) \
    -v $(pwd)/../:/home/developer/S-GAIL_ZK \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -p 8888-8899:8888 \
    -p 6006-6010:6006 \
    -p 5900-5902:5900 \
    --name $(id -u -n)-sgail_zlatko \
    ${IMAGE_NAME} bash
