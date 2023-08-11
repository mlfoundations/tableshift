# Benchmarking Guide

`tableshift` seeks to support a wide range of supervised learning algorithms. Benchmark data is available in a variety of formats used by common learning frameworks (Pandas `DataFrame`s, Torch `DataLoader`s, and even Ray Datasets) and is serializable to disk so that preprocessing need only happen once.

This page provides a walkthrough for bennchmarking an algorithm on the TableShift benchmark.

## Setup

If you haven't already, follow the installation instructions on the main README to set up your environment.

As a check of your setup, make sure that you're able to run the `examples/run_expt.py` script.

## Selecting a Dataset

There are 15 datasets in the TableShift benchmark. All of the datasets are publicly accessible (although a few require public credentialized access). Instructions for accessing these datasets are in the `docs` directory of this repo or on the TableShift website [datasets](https://tableshift.org/datasets.html) page.

To begin with, we recommend starting with a public dataset (those marked as "Public" on the [datasets](https://tableshift.org/datasets.html) page). If you decide to use a public credentialized dataset instead, follow the instructions on the [datasets](https://tableshift.org/datasets.html) page to access the data, download any data file(s), and place them in the TableShift cache directory (by default, this will be `tableshift/tmp`).

## Running An Experiment

TableShift provides a simple interface to the preprocessed datasets, but you have the flexibility to implement training algorithm(s) however you like.

In order to access a dataset with TableShift, simply insert the following lines into your Python script:

``` 
from tableshift import get_dataset
experiment = "diabetes_readmission"  # can be any TableShift dataset name

dset = get_dataset(experiment, cache_dir)

# For Pandas DataFrames
X, y, _, _ = dset.get_pandas("train")
X_test_id, y_test_id, _, _ = dset.get_pandas("test")
X_test_ood, y_test_ood, _, _ = dset.get_pandas("ood_test")

# PyTorch DataLoader
train_loader = dset.get_dataloader("train", batch_size=1024)
id_test_loader = dset.get_dataloader("test", batch_size=1024)
ood_test_loader = dset.get_dataloader("ood_test", batch_size=1024)
```

Using either the Pandas or PyTorch interfaces uses the same underlying dataset on disk, with the same data splits and preprocessing applied -- so you can benchmark algorithms using both sources and reliably compare the results.

## The Full Benchmark

To compute your results on the full benchmark, repeat the above steps for each of the 15 datasets in the TableShift benchmark. Note that the only thing that needs to change for your runs is the name of the parameter provided to `get_dataset`.

## Caching Datasets

The above code will always check to see if the raw data sources are available on disk, and otherwise will attempt to download public data sources. However, preprocessing is repeated each time you instantiate a dataset. This can be slow and expensive for some datasets.

In order to save the preprocessed data to disk, TableShift provides functionality to cache datasets to sharded files on disk.

To cache a dataset, you can either use the caching script at `scripts/cache_task.py`, or cache it yourself, possibly as part of your training pipeline.

Caching is as simple as:

``` 
dset = get_dataset(experiment, cache_dir)
dset.to_sharded(rows_per_shard=4096, file_type="csv")
```

This will create a set of numbered CSV files in the directory specified in `dset.cache_dir` (`tableshift/tmp` by default). Each split (train, validation, etc.) is cached in a separate set of files.

**Loading cached datasets from disk:** Once you've cached a dataset, simply set the `use_cached=True` flag when fetching a dataset:

``` 
dset = get_dataset(experiment, cache_dir, use_cached=True)
```
