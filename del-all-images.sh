#!/bin/bash

# set -ex

help_msg(){
    echo "this script should run on the n-adx-recall-2"
    echo "delete a docker image on the all machines (n-adx-recall-3, n-adx-recall-4)"
    echo "example:"
    echo "     ${0} <the tag or imageID of docker image wanted>"
    echo "     -h, --help: print help message."
}
case "$1" in
  -h|--help)
    help_msg # calling function start()
    exit 1
    ;;
esac

# run on the n-adx-recall-2 server
docker image rm -f $1
echo "remove local image."

for LINE in $(kubectl get node --selector='!node-role.kubernetes.io/master' -o name | awk -F/ '{print $2}'); do
  ssh $LINE "docker image rm $1"
  echo "$LINE is finished."
done

