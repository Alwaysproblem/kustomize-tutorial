#!/bin/bash

set -ex

help_msg(){
    echo "this script should run on the n-adx-recall-2 and run "
    echo "synchronize a docker image to all machine (n-adx-recall-3, n-adx-recall-4) with nfs and ssh"
    echo "example:"
    echo "     ${0} <the tag of docker image wanted>"
    echo "     -h, --help: print help message."
}

case "$1" in
  -h|--help)
    help_msg # calling function start()
    exit 1
    ;;
esac

# run on the n-adx-recall-2 server
# docker image build -t $1 -f $2 ..
docker save $1 > sync.tar
sudo mv sync.tar /data02/
ssh n-adx-recall-3 "docker load < /data02/sync.tar"
ssh n-adx-recall-4 "docker load < /data02/sync.tar"

sudo rm -rf /data02/sync.tar
