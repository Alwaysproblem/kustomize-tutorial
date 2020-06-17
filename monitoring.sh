#!/bin/bash
if [[ -f "train.yaml" ]]; then
    MPI_JOB_NAME=$(cat train.yaml | grep -E "^  name:(.*?)" | awk '/name:/ {print $2}')
else
    MPI_JOB_NAME=$1
fi

PODNAME=$(kubectl get pods -l mpi_job_name=${MPI_JOB_NAME},mpi_role_type=launcher -o name)
kubectl attach ${PODNAME}
