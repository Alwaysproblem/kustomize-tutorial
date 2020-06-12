# Tensorflow model server (TFServing)

## Modify and check all config files or yaml files

- check pv mounted path
  
  the default mounted type is nfs youcan change in the pv yaml file.

- check pvc config file

    **Note that the model is from [this repo](https://github.com/Alwaysproblem/TFServing-setup-review)**

- check deployment file
- check sevice file

    default port is 30020 -> grpc, 30021 -> post
    specify in the service yaml

    **Note that because of no export host IP, please access model server with nodeIP and port**

- the help info of the tensorflow_model_server

    ```markdown
    usage: tensorflow_model_server
    Flags:

        --port=8500
        int32
        Port to listen on for gRPC API

        --grpc_socket_path=""
        string
        If non-empty, listen to a UNIX socket for gRPC API on the given path.
        Can be either relative or absolute path.

        --rest_api_port=0
        int32
        Port to listen on for HTTP/REST API. If set to zero HTTP/REST API will not be exported.
        This port must be different than the one specified in --port.

        --rest_api_num_threads=64
        int32
        Number of threads for HTTP/REST API processing.
        If not set, will be auto set based on number of CPUs.

        --rest_api_timeout_in_ms=30000
        int32
        Timeout for HTTP/REST API calls.

        --enable_batching=false
        bool
        enable batching

        --allow_version_labels_for_unavailable_models=false
        bool
        If true, allows assigning unused version labels to models that are not available yet.

        --batching_parameters_file=""
        string
        If non-empty, read an ascii BatchingParameters protobuf from the supplied file name
        and use the contained values instead of the defaults.

        --model_config_file=""
        string
        If non-empty, read an ascii ModelServerConfig protobuf from the supplied file name, and serve the models in that file.
        This config file can be used to specify multiple models to serve
        and other advanced parameters including non-default version policy.
        (If used, --model_name, --model_base_path are ignored.)

        --model_config_file_poll_wait_seconds=0
        int32
        Interval in seconds between each poll of the filesystemfor model_config_file.
        If unset or set to zero, poll will be done exactly once and not periodically.
        Setting this to negative is reserved for testing purposes only.

        --model_name="default"
        string
        name of model (ignored if --model_config_file flag is set)

        --model_base_path=""
        string
        path to export (ignored if --model_config_file flag is set, otherwise required)

        --max_num_load_retries=5
        int32
        maximum number of times it retries loading a model after the first failure, before giving up.
        If set to 0, a load is attempted only once. Default: 5

        --load_retry_interval_micros=60000000
        int64
        The interval, in microseconds, between each servable load retry.
        If set negative, it doesn't wait. Default: 1 minute

        --file_system_poll_wait_seconds=1
        int32
        Interval in seconds between each poll of the filesystem for new model version.
        If set to zero poll will be exactly done once and not periodically.
        Setting this to negative value will disable polling entirely causing ModelServer
        to indefinitely wait for a new model at startup. Negative values are reserved for testing purposes only.

        --flush_filesystem_caches=true
        bool
        If true (the default), filesystem caches will be flushed after the initial load of all servables,
        and after each subsequent individual servable reload (if the number of load threads is 1).
        This reduces memory consumption of the model server,
        at the potential cost of cache misses if model files are accessed after servables are loaded.

        --tensorflow_session_parallelism=0
        int64
        Number of threads to use for running a Tensorflow session.
        Auto-configured by default.Note that this option is ignored if --platform_config_file is non-empty.

        --tensorflow_intra_op_parallelism=0
        int64
        Number of threads to use to parallelize the executionof an individual op.
        Auto-configured by default.Note that this option is ignored if --platform_config_file is non-empty.

        --tensorflow_inter_op_parallelism=0
        int64
        Controls the number of operators that can be executed simultaneously.
        Auto-configured by default.
        Note that this option is ignored if --platform_config_file is non-empty.

        --ssl_config_file=""
        string
        If non-empty, read an ascii SSLConfig protobuf from the supplied file name and set up a secure gRPC channel

        --platform_config_file=""
        string
        If non-empty, read an ascii PlatformConfigMap protobuf from the supplied file name,
        and use that platform config instead of the Tensorflow platform. (If used, --enable_batching is ignored.)

        --per_process_gpu_memory_fraction=0.000000
        float
        Fraction that each process occupies of the GPU memory space the value is between 0.0 and 1.0 (with 0.0 as the default)
        If 1.0, the server will allocate all the memory when the server starts,
        If 0.0, Tensorflow will automatically select a value.

        --saved_model_tags="serve"
        string
        Comma-separated set of tags corresponding to the meta graph def to load from SavedModel.

        --grpc_channel_arguments=""
        string
        A comma separated list of arguments to be passed to the grpc server.
        (e.g. grpc.max_connection_age_ms=2000)

        --enable_model_warmup=true
        bool
        Enables model warmup, which triggers lazy initializations (such as TF optimizations) at load time,
        to reduce first request latency.

        --version=false
        bool
        Display version

        --monitoring_config_file=""
        string
        If non-empty, read an ascii MonitoringConfig protobuf from the supplied file name

        --remove_unused_fields_from_bundle_metagraph=true
        bool
        Removes unused fields from MetaGraphDef proto message to save memory.

        --use_tflite_model=false
        bool
        EXPERIMENTAL;
        CAN BE REMOVED ANYTIME!
        Load and use TensorFlow Lite model from `model.tflite` file in SavedModel directory
        instead of the TensorFlow model from `saved_model.pb` file.
    ```

## Test

1. start model server

    ```bash
    cd k8s-hvd/Serving
    kubectl apply -f k8s-serving
    # check all with
    kubectl get all -o wide
    ```

2. sending a few request data samples for test through POST(HTTP protocol)

    ```bash
    # curl -d '{"instances": [[1.0, 2.0]]}' -X POST http://K8S-NODE:PORT/v1/models/Toy:predict
    curl -d '{"instances": [[1.0, 2.0]]}' -X POST http://n-adx-recall-4:30021/v1/models/Toy:predict
    # {
    # "predictions": [[0.990161777]
    # ]
    # }
    ```

## online Serving

- only modify the configuration file for tensorflow model server.

    **Note that these directories that `batch`, `config`, `monitor`, `save` is for model server**

    1. `config` includes the config file and version control file
    2. `batch` includes the parameters about inference
    3. `monitor` includes the info about running model server
    4. `save` includes the model local files from training procedure.

<!-- TODO: using configMap to config volumes and secrets! -->