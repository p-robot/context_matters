#!/usr/bin/env python3
"""
Module to load data for use in running FMD outbreaks

Available datasets:
    circular_3km_data_n4000_seed12.csv: 4000 farms randomly distributed
"""

import pandas as pd
from os.path import join, dirname

circ3km = pd.read_csv(join(dirname(__file__), \
    'circular_3km_data_n4000_seed12.csv'))
