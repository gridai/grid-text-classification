# Grid Demo | Text Classification

In this demo example, you'll train a text classification model using [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning), [transformers](https://github.com/huggingface/transformers), and [datasets](https://github.com/huggingface/datasets)!

The full tutorial can be found in the Grid documentation [here](https://docs.grid.ai/examples/nlp/text-classification). 

If you haven't already set up the Grid CLI, follow this [1 minute guide](https://docs.grid.ai/start-here/typical-workflow-cli-user) on how to install the Grid CLI.

**TLDR:** 
`pip install lightning-grid --upgrade`

`grid login`

## Overview 
This example involves three steps: 
1. Downloading the data
2. Uploading the data to Grid using Grid datastores
3. Training the `train.py` script using `grid run`

## Upload the data to Grid

For this example we use the [Lightning Flash](https://lightning-flash.readthedocs.io/en/latest/?badge=latest) IMDB dataset.

```
grid datastore create --source https://pl-flash-data.s3.amazonaws.com/imdb.zip --name imdb-ds
```

When the datastore upload is complete, check the status of the datastore with `grid datastore list`. 
**Wait until Status of datastore shows as Succeeded before moving to the next step.**

## Submit a training run with Grid

Training Parameters
Here are the parameters we'll specify to `grid run`:

**Grid flags:**
1. **--instance_type:** defines number of GPUs and memory
2. **--gpus:** the number of GPUs per experiment
3. **--datastore_name:** the name of the datastore (created above) that you'd like to attach to this training run
4. **--datastore_version:** the version of the datatstore to attach to this training run (defaults to 1)
6. **--grid_disk_size:** the disk size in GB to allocate to each node in the cluster

Then we'll specify the script we're using to train our model followed by the script arguments. 

**Script:** `src/train.py`

These are the arguments defined by the `train.py` script:

**Script arguments:**
1. train_file
2. valid_file
3. test_file
4. max_epochs

Cool! Now we can spin up a Grid run.

**Submit the command below to train a run on a single GPU:** 

```
grid run \
    --name imdb-demo \
    --gpus 1 \
    --instance_type p3.2xlarge \
    --datastore_name imdb-ds \
    --disk_size 500 \
      train.py \
    --gpus 1  \
    --train_file /datastores/imdb-ds/train.csv \
    --valid_file /datastores/imdb-ds/valid.csv  \
    --test_file /datastores/imdb-ds/test.csv \
    --max_epochs 1
```
You can use the `grid status` command to check on the status of the run. To view progess in the Grid UI, use `grid view`. 



