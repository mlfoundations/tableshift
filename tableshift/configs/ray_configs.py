import os

_DEFAULT_RAY_TMP_DIRS = ("/projects/grail/jpgard/ray-scratch",
                         "/data1/home/jpgard/ray-scratch",
                         "/gscratch/scrubbed/jpgard/ray-scratch")
_DEFAULT_RAY_LOCAL_DIRS = ("/projects/grail/jpgard/ray-results",
                           "/data1/home/jpgard/ray-results",
                           "/gscratch/scrubbed/jpgard/ray-results")


def get_default_ray_tmp_dir():
    """Check if any of the default ray tmp dirs exist; if they do, use them."""
    for dirpath in _DEFAULT_RAY_TMP_DIRS:
        if os.path.exists(dirpath):
            ray_tmp_dir = dirpath
            print(
                f"[INFO] detected directory {dirpath}; "
                f"setting this to ray temporary directory.")
            return ray_tmp_dir
    return None


def get_default_ray_local_dir():
    """Check if any of the default ray local dirs exist; if they do, use them."""

    for dirpath in _DEFAULT_RAY_LOCAL_DIRS:
        if os.path.exists(dirpath):
            ray_local_dir = dirpath
            print(
                f"[INFO] detected directory {dirpath}; "
                f"setting this to ray local directory.")
            return ray_local_dir
    return None
