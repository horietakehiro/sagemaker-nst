#!/bin/bash

function daemonize {
    while test 1 -eq 1;
    do
        sleep 60
    done
}

exec jupyter lab --config /config/jupyter_notebook_config.py &

daemonize
