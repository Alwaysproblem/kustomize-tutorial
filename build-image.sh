#!/bin/bash

set -ex

help_msg(){
    echo "this script should run on the n-adx-recall-2 and run "
    echo "build a docker image with dockerfile"
    echo "example:"
    echo "     ${0} <the tag of docker image wanted> <the dockerfile name for building a image>"
    echo "     -h, --help: print help message."
}

case "$1" in
  -h|--help)
    help_msg # calling function start()
    exit 1
    ;;
esac

CURRENT=$(pwd)
# BASE="$(dirname "$(dirname "$CURRENT")")"
echo $CURRENT

docker build -t $1 -f $2 $CURRENT