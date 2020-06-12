# Training prodecdure 1.0 (simplify with kustomize)

## Copy the file from [CTR-prediction](https://git.op-mobile.opera.com/adalgo/opera-ads-research/tree/distribute-training-template)

- downlaod from gitlab

  ```bash
  $ git clone -b distribute-training-template https://git.op-mobile.opera.com/adalgo/opera-ads-research.git
  $ pushd `pwd` && rm -rf .git/ && popd
  $ mv opera-ads-research <your project name>
  ```

## Training

**Note: there is assumption that you already handle with the path of dataset, logs, and model saved.**

- requirement
  - nfs (mounted on /data02)
  - s3fs (mounted on /home/sdev/s3mount)
  - docker (version >= 19.03)
  - kubenetes (version >= 1.14)
  - kustomize (version 3.6.1)
  - mpi-operator

- modify source code `model.py`
  - import horovod and initiate
  - if there are GPUs, disable use all graphic memory
  - wrap the optimizer
  - add horovod callback
  - shard the input data (if you want train all data.)
  - leave all the stdout and saving stage to worker 0

- modify `dockerfile.dev`
  - setup your own envionment on base image
  - copy your code

- test on local machine

  ```bash
  $ docker build -t hvd:test -f dockerfile.dev .
  $ docker run --privileged --rm -it -v $(dirname `pwd`)/tfRecords:/data -v `pwd`/logs:/logs -v `pwd`/models:/models hvd:test /bin/bash
  # in docker.
  root@db72f7e8f23c:/examples# horovodrun -np 4 -H localhost:4 python model.py
  # after local test please remove the docker image
  $ docker image rm hvd:test
  ```

- build docker image and synchronize it (only synchronize to n-adx-recall-3 and n-adx-recall-4)

  ```bash
  # check nfs and s3fs mount working fine.
  $ bash build-image.sh horovod:training dockerfile.dev # this image name will be used in image.yaml utilized in k8s.
  $ bash sync-image.sh horovod:training
  ```

- kustomize base training yaml file
  - edit the file of `kustomize/Training/overlays/plugin`
  - edit image name and tag`image.yaml`
  - edit the prefix and surfix in `prefixsurfix.yaml` to identify the job you submitted
  - edit resource, volumne, replicas, commands and args command in rules [Json6902](https://tools.ietf.org/html/rfc6902)
  - kutomize it

    ```bash
    kustomize build kustomize/Training/overlays/plugin > train.yaml
    ```

- before training
  - create new directory and check data-in directory

    ```bash
    $ ls /home/sdev/s3mount/yongxi/training/ProcessedDataset/
    $ mkdir -p /home/sdev/s3mount/yongxi/training/models/cvr
    $ mkdir -p /home/sdev/s3mount/yongxi/TBlogs/cvr/DeepFM
    ```

  - train

    ```bash
    $ kubectl apply -f train.yaml
    ```

  - monitor

    ```bash
    # if train.yaml file exsist
    bash monitoring.sh
    # else
    bash monitoring.sh <lancher pod name>
    ```
