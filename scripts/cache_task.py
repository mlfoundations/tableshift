"""
Cache a tabular dataset.

Usage:
    python scripts/cache_task.py --experiment heloc
"""
import argparse
import logging

from tableshift import get_dataset
from tableshift.core.utils import make_uid

LOG_LEVEL = logging.DEBUG

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=LOG_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')


def _cache_experiment(experiment: str, cache_dir,
                      overwrite: bool,
                      no_domains_to_subdirectories: bool):
    dset = get_dataset(experiment, cache_dir, initialize_data=False)
    if dset.is_cached() and (not overwrite):
        uid = make_uid(dset.name, dset.splitter)
        logging.info(f"dataset with uid {uid} is already cached; skipping")

    else:
        domains_to_subdirectories = not no_domains_to_subdirectories
        logging.info(
            f"domains_to_subdirectories is {domains_to_subdirectories}")
        dset._initialize_data()
        dset.to_sharded(domains_to_subdirectories=domains_to_subdirectories)
    return


def main(cache_dir,
         experiment,
         overwrite: bool,
         no_domains_to_subdirectories: bool = False,
         domain_shift_experiment=None):
    assert (experiment or domain_shift_experiment) and \
           not (experiment and domain_shift_experiment), \
        "specify either experiment or domain_shift_experiment, but not both."

    cache_kwargs = {
        "cache_dir": cache_dir,
        "overwrite": overwrite,
        "no_domains_to_subdirectories": no_domains_to_subdirectories,
    }
    logging.debug(f"cache_kwargs is: {cache_kwargs}")
    if experiment:
        _cache_experiment(experiment, **cache_kwargs)
        print("caching tasks complete!")
        return

    print("caching tasks complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="tmp",
                        help="Directory to cache raw data files to.")
    parser.add_argument("--domain_shift_experiment", "-d",
                        help="Experiment to run. Overridden when debug=True."
                             "Example value: 'physionet_set'.")
    parser.add_argument("--no_domains_to_subdirectories",
                        action="store_true",
                        help="If set, domains will NOT be written to separate"
                             "subdirectories. For example, instead of writing files to"
                             "/train/1/train_1.csv where the second level is "
                             "the domain value, they will be written to "
                             "/train/train_1.csv and not split by the domain"
                             "value. Useful when using thresholding.")
    parser.add_argument("--experiment",
                        help="Experiment to run. Overridden when debug=True."
                             "Example value: 'adult'.")
    parser.add_argument("--overwrite", action="store_true", default=False)

    args = parser.parse_args()
    main(**vars(args))
