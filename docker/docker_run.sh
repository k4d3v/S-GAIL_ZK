IMAGE_NAME="sgail_zlatko"

docker run --rm -it \
    --privileged \
    --net=host\
    -e DISPLAY=$DISPLAY\
    -e LOCAL_UID=$(id -u $USER) \
    -e LOCAL_GID=$(id -g $USER) \
    -e USE_VNC='true' \
    -e CHANGE_USER='true' \
    -e SCREEN_RESOLUTION=1920x1800x24 \
    -e VNC_PASSWORD=passpass \
    -e DISPLAY=":5" \
    -v $(pwd)/../:/home/developer/S-GAIL_ZK \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -p 8888-8899:8888 \
    -p 6006-6010:6006 \
    -p 5900-5902:5900 \
    --name $(id -u -n)-sgail_zlatko \
    ${IMAGE_NAME} bash
