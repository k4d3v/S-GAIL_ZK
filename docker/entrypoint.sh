#!/bin/bash

# # inside docker script

# # 0. generate xorg.conf if not copied
# [ ! -e /etc/X11/xorg.conf ] && nvidia-xconfig -a --virtual=$SCREEN_RESOLUTION --allow-empty-initial-configuration --enable-all-gpus --busid $BUSID

# # 1. launch X server
# Xorg :0 &
# sleep 1  # wait for the server gets ready

# # 2. start x11 and vnc connection
# x11vnc -display :0 -passwd $VNC_PASSWORD -forever &
# sleep 1  # wait for the server gets ready

# # 3. start simulator
# export DISPLAY=:0
# sleep 1

USER_ID=${LOCAL_UID:-9001}
GROUP_ID=${LOCAL_GID:-9001}

usermod -u $USER_ID -o -d /home/developer -m developer
groupmod -g $GROUP_ID developer

exec /usr/sbin/gosu developer "$@"