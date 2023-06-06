import collections
import datetime
import logging
import os
import re
import subprocess
import urllib.parse
from itertools import islice

import pandas as pd
import requests
import xport.v56

from .splitter import Splitter, DomainSplitter


def initialize_dir(dir: str):
    """Create a directory if it does not exist."""
    if not os.path.exists(dir):
        os.makedirs(dir)


def basename_from_url(url: str) -> str:
    parsed_url = urllib.parse.urlparse(url)
    fname = os.path.basename(parsed_url.path)
    return fname


def download_file(url: str, dirpath: str, if_not_exist=True,
                  dest_file_name=None):
    """Download file to the specified directory."""
    if dest_file_name:
        fname = dest_file_name
    else:
        fname = basename_from_url(url)
    fpath = os.path.join(dirpath, fname)

    if os.path.exists(fpath) and if_not_exist:
        # Case: file already exists; skip it.
        logging.debug(f"not downloading {url}; exists at {fpath}")

    else:
        initialize_dir(dirpath)
        logging.debug(f"downloading {url} to {fpath}")
        with open(fpath, "wb") as f:
            f.write(requests.get(url).content)

    return fpath


def read_xpt(fp) -> pd.DataFrame:
    assert os.path.exists(fp), "file does not exist %s" % fp
    with open(fp, "rb") as f:
        obj = xport.v56.load(f)
    # index into SAS structure, assuming there is only one dataframe
    assert len(tuple(obj._members.keys())) == 1
    key = tuple(obj._members.keys())[0]
    ds = obj[key]
    # convert xport.Dataset to pd.DataFrame
    columns = [ds[c] for c in ds.columns]
    df = pd.DataFrame(columns).T
    del ds
    return df


def run_in_subprocess(cmd):
    logging.info(f"running {cmd}")
    res = subprocess.run(cmd, shell=True)
    logging.debug(f"{cmd} returned {res}")
    return res


def sliding_window(iterable, n):
    """ From https://docs.python.org/3/library/itertools.html"""
    # sliding_window('ABCDEFG', 4) --> ABCD BCDE CDEF DEFG
    it = iter(iterable)
    window = collections.deque(islice(it, n), maxlen=n)
    if len(window) == n:
        yield tuple(window)
    for x in it:
        window.append(x)
        yield tuple(window)


def make_uid(name: str, splitter: Splitter, replace_chars="*/:'$!") -> str:
    """Make a unique identifier for an experiment."""
    uid = name
    if isinstance(splitter, DomainSplitter) and splitter.is_explicit_split():
        attrs = {'domain_split_varname': splitter.domain_split_varname,
                 'domain_split_ood_value': ''.join(
                     str(x) for x in splitter.domain_split_ood_values)}
        if splitter.domain_split_id_values:
            attrs['domain_split_id_values'] = ''.join(
                str(x) for x in splitter.domain_split_id_values)
        uid += ''.join(f'{k}_{v}' for k, v in attrs.items())
    elif isinstance(splitter, DomainSplitter) and splitter.is_threshold_split():
        uid += f'{splitter.domain_split_varname}gt{splitter.domain_split_gt_thresh}'

    # if any slashes exist, replace with periods.
    for char in replace_chars:
        uid = uid.replace(char, '.')
    # Max path length on some OSs is 255.
    return uid[:240]


def timestamp_as_int() -> int:
    """Helper function to get the current timestamp as int."""
    dt = datetime.datetime.now()
    return int(dt.strftime("%Y%m%d%H%M%S"))


ILLEGAL_CHARS_REGEX = '[\\[\\]{}.:<>/,"]'


def contains_illegal_chars(s: str) -> bool:
    if re.search(ILLEGAL_CHARS_REGEX, s):
        return True
    else:
        return False


def sub_illegal_chars(s: str) -> str:
    return re.sub(ILLEGAL_CHARS_REGEX, "", s)


def convert_64bit_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Convert float64 or int64 columns to float/int32.

    Not all numpy dtypes are compatible with 64-bit precision."""
    int64_float64_cols = list(df.select_dtypes(include=['float64', 'int64']))
    df[int64_float64_cols] = df[int64_float64_cols].astype('float32')
    return df
