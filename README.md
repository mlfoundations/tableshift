![status](https://github.com/mlfoundations/tableshift/actions/workflows/python-package-conda.yml/badge.svg)
![status](https://github.com/mlfoundations/tableshift/actions/workflows/run-example-script.yml/badge.svg)
![status](https://github.com/mlfoundations/tableshift/actions/workflows/docker.yml/badge.svg)

![tableshift logo](img/tableshift.png)

# TableShift

TableShift is a benchmarking library for machine learning with tabular data under distribution shift.

You can read more about TableShift at [tableshift.org](https://tableshift.org/index.html) or read the full paper (published in NeurIPS 2023 Datasets & Benchmarks Track) on [arxiv](https://arxiv.org/abs/2312.07577).

If you find an issue, please file a GitHub [issue](https://github.com/mlfoundations/tableshift/issues/new/choose).

# Quickstart

**Environment setup:** We recommend the use of docker with TableShift. Our dataset construction and model pipelines have a diverse set of dependencies that included non-Python files required to make some libraries work. As a result, we recommend you use the provided Docker image for using the benchmark, and suggest forking this Docker image for your own development.

```bash 
# fetch the docker image
docker pull ghcr.io/jpgard/tableshift:latest

# run it to test your setup; this automatically launches examples/run_expt.py
docker run ghcr.io/jpgard/tableshift:latest --model xgb

# optionally, use the container interactively
docker run -it --entrypoint=/bin/bash ghcr.io/jpgard/tableshift:latest

```

**Conda:** We recommend using Docker with TableShift when running training or using any of the pretrained modeling code, as the libraries used for training contain a complex and subtle set of dependencies that can be difficult to configure outside Docker. However, Conda might provide a more lightweight environment for basic development and exploration with TableShift, so we describe how to set up Conda here. 

To create a conda environment, simply clone this repo, enter the root directory, and run the following commands to create and test a local execution environment:

```bash
# set up the environment
conda env create -f environment.yml
conda activate tableshift
# test the install by running the training script
python examples/run_expt.py
```

The final line above will print some detailed logging output as the script executes. When you see `training completed! test accuracy: 0.6221` your environment is ready to go! (Accuracy may vary slightly due to randomness.)

**Accessing datasets:** If you simply want to load and use a standard version of
one of the public TableShift datasets, it's as simple as:

```python
from tableshift import get_dataset

dataset_name = "diabetes_readmission"
dset = get_dataset(dataset_name)
```

The full list of identifiers for all available datasets is below; simply swap any of these for `dataset_name` to access the relevant data.

If you would like to use a dataset *without* a domain split, replace `get_dataset()` with `get_iid_dataset()`.

The call to `get_dataset()` returns a `TabularDataset` that you can use to
easily load tabular data in several formats, including Pandas DataFrame and
PyTorch DataLoaders:

```python
# Fetch a pandas DataFrame of the training set
X_tr, y_tr, _, _ = dset.get_pandas("train")

# Fetch and use a pytorch DataLoader
train_loader = dset.get_dataloader("train", batch_size=1024)

for X, y, _, _ in train_loader:
    ...
```

For all TableShift datasets, the following splits are
available: `train`, `validation`, `id_test`, `ood_validation`, `ood_test`.

For IID datasets (those without a domain split) these splits are available: `train`, `validation`, `test`.

There is a complete example of a training script in `examples/run_expt.py`.

# Benchmark Dataset Availability

*tl;dr: if you want to get started exploring ASAP, use datasets marked as "
public" below.*

All of the datasets used in the TableShift benchmark are either publicly available or provide open credentialized
access.
The datasets with open credentialized access require signing a data use agreement; as a result,
some datasets must be manually fetched and stored locally. TableShift makes this process as simple as possible.

A list of datasets, their names in TableShift, and the corresponding access
levels are below. The string identifier is the value that should be passed as the `experiment` parameter
to `get_dataset()` or the `--experiment` flag of `run_expt.py` and other training scripts.

| Dataset                 | String Identifier         | Availability                                                                                                 | Source                                                                                                                 |
|-------------------------|---------------------------|--------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| Voting                  | `anes`                    | Public Credentialized Access ([source](https://electionstudies.org))                                         | [American National Election Studies (ANES)](https://electionstudies.org)                                               |
| ASSISTments             | `assistments`             | Public                                                                                                       | [Kaggle](https://www.kaggle.com/datasets/nicolaswattiez/skillbuilder-data-2009-2010)                                   |
| Childhood Lead          | `nhanes_lead`             | Public                                                                                                       | [National Health and Nutrition Examination Survey (NHANES)](https://www.cdc.gov/nchs/nhanes/index.htm)                 |
| College Scorecard       | `college_scorecard`       | Public                                                                                                       | [College Scorecard](http://collegescorecard.ed.gov)                                                                    |
| Diabetes                | `brfss_diabetes`          | Public                                                                                                       | [Behavioral Risk Factor Surveillance System (BRFSS)](https://www.cdc.gov/brfss/index.html)                             |
| Food Stamps             | `acsfoodstamps`           | Public                                                                                                       | [American Community Survey](https://www.census.gov/programs-surveys/acs) (via [folktables](http://folktables.org)      |
| HELOC                   | `heloc`                   | Public Credentialized Access ([source](https://community.fico.com/s/explainable-machine-learning-challenge)) | [FICO](https://community.fico.com/s/explainable-machine-learning-challenge)                                            |
| Hospital Readmission    | `diabetes_readmission`    | Public                                                                                                       | [UCI](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)                           |
| Hypertension            | `brfss_blood_pressure`    | Public                                                                                                       | [Behavioral Risk Factor Surveillance System (BRFSS)](https://www.cdc.gov/brfss/index.html)                             |
| ICU Length of Stay      | `mimic_extract_los_3`     | Public Credentialized Access ([source](https://mimic.mit.edu/docs/gettingstarted/))                          | [MIMIC-iii](https://physionet.org/content/mimiciii/) via [MIMIC-Extract](https://github.com/MLforHealth/MIMIC_Extract) |
| ICU Mortality           | `mimic_extract_mort_hosp` | Public Credentialized Access ([source](https://mimic.mit.edu/docs/gettingstarted/))                          | [MIMIC-iii](https://physionet.org/content/mimiciii/) via [MIMIC-Extract](https://github.com/MLforHealth/MIMIC_Extract) |
| Income                  | `acsincome`               | Public                                                                                                       | [American Community Survey](https://www.census.gov/programs-surveys/acs) (via [folktables](http://folktables.org)      |
| Public Health Insurance | `acspubcov`               | Public                                                                                                       | [American Community Survey](https://www.census.gov/programs-surveys/acs) (via [folktables](http://folktables.org)      |
| Sepsis                  | `physionet`               | Public                                                                                                       | [Physionet](https://physionet.org/content/challenge-2019/)                                                             |
| Unemployment            | `acsunemployment`         | Public                                                                                                       | [American Community Survey](https://www.census.gov/programs-surveys/acs) (via [folktables](http://folktables.org)      |

Note that details on the data source, which files to load, and the feature
codings are provided in the TableShift source code for each dataset and data
source (see `data_sources.py` and the `tableshift.datasets` module).

For additional, non-benchmark datasets (possibly with only IID splits, not a distribution shift),
see `tableshift.configs.non_benchmark.configs.py`

# Dataset Details

More information about the tasks, datasets, splitting variables, data sources, and motivation are available in the
TableShift paper; we provide a summary below.

| Task                    | Target                                                       | Shift                       | Domain   | Baseline | Total Observations |
|-------------------------|--------------------------------------------------------------|-----------------------------|----------|----------|--------------------|
| ASSISTments             | Next Answer Correct                                          | School                      | &#10003; | -34.5%   | 2,667,776          |
| College Scorecard       | Low Degree Completion Rate                                   | Institution Type            | &#10003; | -11.2%   | 124,699            |
| ICU Mortality  | ICU patient expires in hospital during current visit         | Insurance Type              | &#10003; | -6.3%    | 23,944             |
| Hospital Readmission    | 30-day readmission of diabetic hospital patients             | Admission source            | &#10003; | -5.9%    | 99,493             |
| Diabetes                | Diabetes diagnosis                                           | Race                        | &#10003; | -4.5%    | 1,444,176          |
| ICU Length of Stay      | Length of stay >= 3 hrs in ICU                               | Insurance Type              | &#10003; | -3.4%    | 23,944             |
| Voting                  | Voted in U.S. presidential election                          | Geographic Region           | &#10003; | -2.6%    | 8280               |
| Food Stamps             | Food stamp recipiency in past year for households with child | Geographic Region           | &#10003; | -2.4%    | 840,582            |
| Unemployment            | Unemployment for non-social security-eligible adults         | Education Level             | &#10003; | -1.3%    | 1,795,434          |
| Income                  | Income >= 56k for employed adults                            | Geographic Region           | &#10003; | -1.3%    | 1,664,500          |
| HELOC              | Repayment of Home Equity Line of Credit loan                 | Est. third-party risk level |          | -22.6%   | 10,459             |
| Public Health Insurance | Coverage of non-Medicare eligible low-income individuals     | Disability Status           |          | -14.5%   | 5,916,565          |
| Sepsis                  | Sepsis onset within next 6hrs for hospital patients          | Length of Stay              |          | -6.0%    | 1,552,210          |
| Childhood Lead          | Blood lead levels above CDC Blood Level Reference Value      | Poverty level               |          | -5.1%    | 27,499             |
| Hypertension            | Hypertension diagnosis for high-risk age (50+)               | BMI Category                |          | -4.4%    | 846,761            |

# A Self-Contained Training Example

A sample training script is located at `examples/run_expt.py`. However, training a scikit-learn model is as simple as:

```python
from tableshift import get_dataset
from sklearn.ensemble import GradientBoostingClassifier

dset = get_dataset("diabetes_readmission")
X_train, y_train, _, _ = dset.get_pandas("train")

# Train
estimator = GradientBoostingClassifier()
trained_estimator = estimator.fit(X_train, y_train)

# Test
for split in ('id_test', 'ood_test'):
    X, y, _, _ = dset.get_pandas(split)
    preds = estimator.predict(X)
    acc = (preds == y).mean()
    print(f'accuracy on split {split} is: {acc:.3f}')
```

The code should output the following:

```  
accuracy on split id_test is: 0.655
accuracy on split ood_test is: 0.619
```

Now, please close that domain gap!

# Non-benchmark datasets

We also have several tabular datasets available in TableShift which are not part of the official TableShift benchmark,
but which still may be useful for tabular data research. We are continuously adding datasets to the package. These
datasets support all of the same functionality provided for the TableShift benchmark datasets, but we did not include
these as an official part of the TableShift benchmark -- they are not an official part of the TableShift package and are
mostly intended for convenience and for our own internal use.

For a list of the non-benchmark datasets, see the file `tableshift.configs.non_benchmark_configs.py`.
