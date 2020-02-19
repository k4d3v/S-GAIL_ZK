#!/usr/bin/env bash
set -e

if "${USE_VNC}"; then
    export DISPLAY=${DISPLAY}

    Xvfb ${DISPLAY} -screen 0 $SCREEN_RESOLUTION &
    sleep 1

    x11vnc -display ${DISPLAY} -passwd $VNC_PASSWORD -forever &
    sleep 1

    icewm-session &
    sleep 1
fi

# setup ros environment
# source "/opt/ros/$ROS_DISTRO/setup.bash"
# source "/home/developer/catkin_ws/devel/setup.bash"

if "${CHANGE_USER}"; then
    USER_ID=${LOCAL_UID:-9001}
    GROUP_ID=${LOCAL_GID:-9001}

    usermod -u $USER_ID -o -d /home/developer -m developer
    groupmod -g $GROUP_ID developer

    CMD="/usr/sbin/gosu developer"
else
    CMD=""
fi

exec $CMD "$@"
