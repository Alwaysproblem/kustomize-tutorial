#!/bin/bash

# set -ex

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
echo "saved image"
sudo mv sync.tar /data02/
echo "synchronizing"


for LINE in $(kubectl get node --selector='!node-role.kubernetes.io/master' -o name | awk -F/ '{print $2}'); do
  ssh $LINE "docker load < /data02/sync.tar"
  echo "$LINE is finished."
done

if [[ -f "/data02/sync.tar" ]]; then
  sudo rm -rf /data02/sync.tar
  echo "removed sync.tar."
fi
