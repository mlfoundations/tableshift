# Accessing Open Credentialized Datasets

This page gives general instructions on using the different types of datasets in TableShift. In particular, it describes the process for configuring a dataset with open credentialized access so that it can be used with TableShift.

*tl;dr: No action is required for public datasets. For credentialized datasets, follow the provided links [here](https://tableshift.org/datasets.html) to obtain access, and download the necessary file(s) to the TableShift cache (`tableshift/tmp` by default).*

### Overview

All TableShift benchmark datasets are available to anyone, but some require action on the users' behalf to obtain access. The TableShift benchmark contains two types of datasets: public datasets (no usage restrictions) and datasets with open credentialized access. Open credentialized access means that access to a dataset is available to anyone, as long as they can provide certain credentials to the dataset maintainers (such as filling out a data use agreement or, in the case of sensitive human subjects data, completing necessary free human subjects training).

Before beginning experiments with a specific benchmark dataset, verify the access level of the dataset. This can be done by checking the paper, the table in our main README in this repo, or the TableShift website. *If a dataset is marked as "Public", no action is required and the TableShift Python API will fetch the data automatically the first time it is used.* (After the first usage, the data will be fetched from a local cache.)

### Accessing an Open Credentialized Dataset

The instructions here are for accessing open credentialized datasets. For instructions on how to access the data files for each individual dataset, check the [datasets](https://tableshift.org/datasets.html) page on the TableShift website. The links to any data use agreement(s) and the specific files used are described for each dataset on that page under "Availability & Access".

To use an open credentialized dataset:
1. **Credentialization:** Complete any credentialization required for the dataset (described on the TableShift [datasets](https://tableshift.org/datasets.html) page).
2. **File Download:** Download the necessary file(s) to the TableShift cache directory. By default, this is located at `tableshift/tmp`, but you can provide another `cache_dir` to the TableShift dataset constructors. No preprocessing or renaming of the files is necessary.

After completing these steps, the dataset should be ready for use in the TableShift benchmark!

### Example: American National Election Survey (ANES)

Here we give a brief example of how to set up a public credentialized dataset, using the American National Election Survey (ANES) as an example.

1. **Credentialization:** As listed on the TableShift [datasets](https://tableshift.org/datasets.html) page and the README of this repo, accessing the ANES data requires registering on the ANES website. Create an account.
2. **File Download:** Access the September 16, 2022 Time Series Cumulative Data File (click "Data Center" > "Time Series Cumulative Data File (1948-2020)" > CSV). Download this file and place it at `tableshift/tmp`.

You can verify your installation by running the following in a Python terminal:

``` 
from tableshift import get_dataset
dset = get_dataset("anes")
```

To access any other public credentialized access dataset in the benchmark, follow the same steps above. Links to access datasets are on the [datasets](https://tableshift.org/datasets.html) page and the README at the root of this repo.