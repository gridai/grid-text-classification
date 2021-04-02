# Grid Demo | Text Classification

In this demo example, you'll train a text classification model using [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning), [transformers](https://github.com/huggingface/transformers), and [datasets](https://github.com/huggingface/datasets)

If you haven't already set up the Grid CLI, follow this [1 minute guide](https://app.gitbook.com/@grid-ai/s/grid-cli/start-here/typical-workflow-cli-user#step-0-install-the-grid-cli) on how to install the Grid CLI.

**TLDR:** 
`pip install lightning-grid --upgrade`

`grid login`

## Overview 
This example involves three steps: 
1. Downloading the data
2. Uploading the data to Grid using Grid datastores
3. Training the `train.py` script using `grid train`

## Download the data

For this example we use the [Lightning Flash](https://lightning-flash.readthedocs.io/en/latest/?badge=latest) IMDB dataset.

```
wget https://pl-flash-data.s3.amazonaws.com/imdb.zip
unzip imdb.zip
```

## Upload data to Grid:
Now that you have some data, let's upload that as a Grid Datastore. 

```
grid datastores create --source imdb --name imdb
```

Here the `--source` represents the path to the data which will be uploaded. The `--name` will be used as the datastore name in Grid. 

When the datastore upload is complete, check the status of the datastore with `grid datastores list`. 
**Wait until the datastore status is `Succeeded` before moving to the next step.**

## Submit a training run with Grid

Training Parameters
Here are the parameters we'll specify to `grid train`:

**Grid flags:**
1. **--grid_instance_type:** defines number of GPUs and memory
2. **--grid_gpus:** the number of GPUs per experiment
3. **--grid_datastore_name:** 
4. **--grid_datastore_version:** 
5. **--grid_datastore_mount_dir:**
6. **--grid_disk_size:**

Then we'll specify the script we're using to train our model followed by the script arguments. 

**Script:** `src/train.py`

These are the arguments defined by the `train.py` script:

**Script arguments:**
1. train_file
2. valid_file
3. test_file
4. max_epochs
5. plugins ddp_sharded

Cool! Now we can spin up a Grid Train run.

**Submit the command below to train a run on a single GPU:** 

```
grid train \
    --grid_name imdb-demo \
    --grid_gpus 1 \
    --grid_instance_type p3.2xlarge \
    --grid_datastore_name imdb \
    --grid_datastore_mount_dir /dataset/imdb \
    --grid_disk_size 500 \
      train.py \
    --gpus 1  \
    --train_file /dataset/imdb/train.csv \
    --valid_file /dataset/imdb/valid.csv  \
    --test_file /dataset/imdb/test.csv \
    --max_epochs 1
```
You can use the `grid status` command to check on the status of the run. To view progess in the Grid UI, use `grid view`. 

**Submit the command below to enable Mixed Precision:**

```
grid train \
    --grid_gpus 1 \
    --grid_instance_type p3.2xlarge \
    --grid_datastore_name "imdb" \
    --grid_datastore_version 1 \
    --grid_datastore_mount_dir /dataset/imdb \
    --grid_disk_size 500 \
      train.py \
    --gpus 1  \
    --train_file /dataset/imdb/train.csv \
    --valid_file /dataset/imdb/valid.csv  \
    --test_file /dataset/imdb/test.csv \
    --max_epochs 1
```

**Submit the command below to train a run on 4 GPUs with [Sharded Training](http://localhost:63342/pytorch-lightning/docs/build/html/advanced/multi_gpu.html#sharded-training):**

```
grid train \
    --grid_gpus 4 \
    --grid_instance_type p3.8xlarge \
    --grid_datastore_name "imdb" \
    --grid_datastore_version 1 \
    --grid_datastore_mount_dir /dataset/imdb \
    --grid_disk_size 500 \
      train.py \
    --gpus 1  \
    --train_file /dataset/imdb/train.csv \
    --valid_file /dataset/imdb/valid.csv  \
    --test_file /dataset/imdb/test.csv \
    --max_epochs 1 \
    --plugins ddp_sharded
```
