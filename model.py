# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# from IPython.display import display
import json
import os
# %%
from datetime import datetime

import numpy as np
import tensorflow as tf
from deepctr.inputs import SparseFeat, VarLenSparseFeat, get_feature_names
from deepctr.models import DeepFM

import horovod.tensorflow.keras as hvd

#%%
# horovod initiation procedure
hvd.init()
num_workers = hvd.size()
worker_index = hvd.rank()
# if there are GPUs, disable use all graphic memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# %%
# sparse = [1, 2, 3, 8, 10, 12, 13, 68, 69, 151, 152, 163, 166, 77]
# varlen = [65, 66, 67, 71, 72, 73, 74, 75, 76, 158, 159]

sparse = [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 
            58, 59, 60, 61, 62, 64, 68, 69, 77, 81, 82, 83, 84, 85, 86, 87,
            90, 92, 98, 100, 101, 102, 103, 104, 151, 152, 160,
            161, 162, 163, 164, 166, 167, 251]

varlen = [51, 52, 53, 54, 55, 56, 57, 63, 65, 66, 67, 71, 72, 73, 
            74, 75, 76, 88, 89, 91, 93, 94, 95, 96, 97, 99, 153,
            154, 155, 156, 157, 158, 159, 165]

sparse_f = [f"f{i}" for i in sparse]
varlen_f = [f"f{i}" for i in varlen]

with open("./config.json") as f:
    sparse_vcab_dic, varlen_vcab_dic,\
         varlen_maxlen_f, len_train, len_valid =\
                 json.load(f)


# %%
col_type = {feat: "int64" for feat in sparse_f}
col_type.update({vfeat: "int64List" for vfeat in varlen_f})
col_type.update({'label': "int64"})
# col_type


# %%
NNconfig_dic = {
    "batchsize": 2048,
    "epochs": 5,
    "buffersize": 4096,
    "embedding_size": 4,
    "lr": 0.001,
    "shuffled": False,
    "dnn_hidden_units": (256, 256, 256,),
    "l2_reg_dnn": 0,
    "l2_reg_embedding": 1e-5,
    "l2_reg_linear": 1e-5,
    "l2_reg_cin": 0,
    "dnn_dropout": 0,
    "dnn_use_bn": False,
    "dnn_activation": 'relu',
    "cin_layer_size": (64, 64, 64, 64, 64),
    "att_layer_num": 3,
    "att_embedding_size": 8,
    "att_head_num": 2,
    "att_res": True,
}

train_path = "/data/train/*"
valid_path = "/data/valid/*"


# %%
def Decode(file_paths: list, col_type: dict, target: str, batchsize: int,
           num_parallel_calls=None, buffer_size=None, block_length=1, cycle_length=None,
           sparses=[], varlens=[], numerics=[],
           globalSparsePara={}, globalVarlenPara={},
           globalNumericPara={}, omit_label = False):
    op_dic = {
            'stringv': tf.io.VarLenFeature(tf.string),
            'string': tf.io.FixedLenFeature((), tf.string),
            'int32': tf.io.FixedLenFeature((), tf.int32),
            'int64': tf.io.FixedLenFeature((), tf.int64),
            'float32': tf.io.FixedLenFeature((), tf.float32),
            'float64': tf.io.FixedLenFeature((), tf.float64),
            'int32List': tf.io.VarLenFeature(tf.int32),
            'int64List': tf.io.VarLenFeature(tf.int64),
            'float32List': tf.io.VarLenFeature(tf.float32),
            'float64List': tf.io.VarLenFeature(tf.float64),
        }
    feature_key_value_pair = {}
    for col in col_type:
        feature_key_value_pair[col] = op_dic[col_type[col]]

    def map_decoder(serialized_example):
        
        sample = tf.io.parse_example(serialized_example, feature_key_value_pair)

        if len(varlens) != 0 and len(globalVarlenPara) != 0:
            for v in varlens:
                sample[v] = tf.sparse.to_dense(sample[v])
                # sample[v].set_shape((tf.newaxis, globalVarlenPara[v]))
        
        y = sample.pop(target)
        if omit_label == True:
            return sample
        else:
            return (sample, y)
        # return sample
    
    files = tf.data.Dataset.list_files(file_paths)
    if cycle_length is not None:
        dataset = files.interleave(lambda x:
                        tf.data.TFRecordDataset(x)\
                            .batch(batchsize).map(map_decoder, num_parallel_calls=num_parallel_calls),
                        cycle_length=cycle_length,
                        block_length=block_length,
                        num_parallel_calls=num_parallel_calls)
    else:
        dataset = files.interleave(lambda x:
                        tf.data.TFRecordDataset(x)\
                            .batch(batchsize).map(map_decoder, num_parallel_calls=num_parallel_calls),
                        block_length=block_length,
                        num_parallel_calls=num_parallel_calls)
    return dataset


# %%
batchsize = NNconfig_dic["batchsize"]
epochs = NNconfig_dic["epochs"]
buffer_size = NNconfig_dic["buffersize"]
num_para = tf.data.experimental.AUTOTUNE
D_train_r = Decode(train_path,
            col_type, 
            target = 'label', 
            batchsize = batchsize,
            block_length=batchsize,
            num_parallel_calls=num_para,
            sparses= sparse_f, 
            varlens=varlen_f,
            globalVarlenPara=varlen_maxlen_f
            )
D_valid_r = Decode(valid_path,
            col_type, 
            target = 'label', 
            batchsize = batchsize, 
            block_length=batchsize,
            num_parallel_calls=num_para,
            sparses= sparse_f, 
            varlens=varlen_f,
            globalVarlenPara=varlen_maxlen_f
            )
# display(dataset)


# %%
if NNconfig_dic["shuffled"] == True:
    D_train = D_train.shuffle(buffer_size)
else:
    pass

if gpus:
    D_train = D_train_r\
                .shard(num_workers, worker_index)\
                .repeat()\
                .apply(
                    tf.data.experimental\
                    .prefetch_to_device('/gpu:0', buffer_size=num_para)) 
    D_valid = D_valid_r\
                .shard(num_workers, worker_index)\
                .repeat()\
                .apply(
                    tf.data.experimental\
                    .prefetch_to_device('/gpu:0', buffer_size=num_para))
else:
    D_train = D_train_r.shard(num_workers, worker_index).repeat().prefetch(buffer_size=num_para)
    D_valid = D_valid_r.shard(num_workers, worker_index).repeat().prefetch(buffer_size=num_para)


# %%
embedding_size = NNconfig_dic["embedding_size"]
sparse_feature_columns = []
varlen_feature_columns = []

sparse_feature_columns = [SparseFeat(feat, sparse_vcab_dic[feat] + 1, 
                                dtype=tf.int64, embedding_dim = embedding_size) for feat in sparse_f]
varlen_feature_columns = [VarLenSparseFeat(SparseFeat(vfeat,  
                            vocabulary_size = varlen_vcab_dic[vfeat] + 1,
                            dtype=tf.int64, embedding_dim = embedding_size), maxlen = varlen_maxlen_f[vfeat]) for vfeat in varlen_f]


# %%
linear_feature_columns, dnn_feature_columns = \
    sparse_feature_columns + varlen_feature_columns, sparse_feature_columns + varlen_feature_columns


# %%
model = DeepFM(linear_feature_columns, dnn_feature_columns,
                dnn_hidden_units=NNconfig_dic["dnn_hidden_units"], 
                l2_reg_dnn=NNconfig_dic["l2_reg_dnn"],
                l2_reg_embedding=NNconfig_dic["l2_reg_embedding"],
                l2_reg_linear=NNconfig_dic["l2_reg_linear"],
                dnn_dropout=NNconfig_dic["dnn_dropout"],
                dnn_use_bn=NNconfig_dic["dnn_use_bn"],
                dnn_activation=NNconfig_dic["dnn_activation"])
NNconfig_dic["model_name"] = "DeepFM"

# model = xDeepFM(linear_feature_columns, dnn_feature_columns,
#                 dnn_hidden_units=NNconfig_dic["dnn_hidden_units"], 
#                 cin_layer_size=NNconfig_dic["cin_layer_size"],
#                 l2_reg_dnn=NNconfig_dic["l2_reg_dnn"],
#                 l2_reg_embedding=NNconfig_dic["l2_reg_embedding"],
#                 l2_reg_linear=NNconfig_dic["l2_reg_linear"],
#                 l2_reg_cin=NNconfig_dic["l2_reg_cin"],
#                 dnn_use_bn=NNconfig_dic["dnn_use_bn"],
#                 dnn_activation=NNconfig_dic["dnn_activation"],
#                 dnn_dropout=NNconfig_dic["dnn_dropout"])
# NNconfig_dic["model_name"] = "xDeepFM"

# model = AutoInt(linear_feature_columns, dnn_feature_columns,
#                 dnn_hidden_units=NNconfig_dic["dnn_hidden_units"], 
#                 l2_reg_dnn=NNconfig_dic["l2_reg_dnn"],
#                 l2_reg_embedding=NNconfig_dic["l2_reg_embedding"],
#                 l2_reg_linear=NNconfig_dic["l2_reg_linear"],
#                 dnn_use_bn=NNconfig_dic["dnn_use_bn"],
#                 dnn_activation=NNconfig_dic["dnn_activation"],
#                 dnn_dropout=NNconfig_dic["dnn_dropout"],
#                 att_layer_num=NNconfig_dic["att_layer_num"],
#                 att_embedding_size=NNconfig_dic["att_embedding_size"],
#                 att_head_num=NNconfig_dic["att_head_num"],
#                )
# NNconfig_dic["model_name"] = "AutoInt"


# %%
opt = tf.keras.optimizers.Adam(learning_rate=NNconfig_dic["lr"])
opt_hvd = hvd.DistributedOptimizer(opt, compression=hvd.Compression.fp16)
NNconfig_dic["optimizer"] = "Adam"


# %%
model.compile(optimizer=opt_hvd, loss=tf.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.AUC()])

log_dir="logs"+ os.path.sep + NNconfig_dic["model_name"] + "_res" + os.path.sep \
              + datetime.now().strftime("%Y%m%d-%H%M%S")
NN_config_path = "logs" + os.path.sep + NNconfig_dic["model_name"] + "_res" + os.path.sep \
              + datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + "NNconfig.json"


if worker_index == 0:
    if not os.path.exists("logs" + os.path.sep + NNconfig_dic["model_name"] + "_res"):
        os.makedirs("logs" + os.path.sep + NNconfig_dic["model_name"] + "_res")
    with open(NN_config_path, "w+") as conf:
        json.dump(NNconfig_dic, conf)

# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch = 3)

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
]

if worker_index == 0:
    model.summary()
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir='/logs', histogram_freq=1))


# %%
model.fit(D_train, epochs=epochs, verbose=1 if worker_index == 0 else 0, validation_data=D_valid,
                    steps_per_epoch=max(len_train // batchsize + 1, num_workers) // num_workers , 
                    validation_steps=max(len_valid // batchsize + 1, num_workers) // num_workers,
                    callbacks = callbacks)

# %%
if worker_index == 0:
    save_path = '/models/save/CTR/1'
    model.save(save_path)
