# Copyright 2018 Databricks, Inc.
import re


VERSION = "blume-v1.2"


def is_release_version():
    return bool(re.match(r"^\d+\.\d+\.\d+$", VERSION))
