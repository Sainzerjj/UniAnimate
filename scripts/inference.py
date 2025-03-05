import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))  
sys.path.append(os.path.join(script_dir, '..'))  
import copy
import json
import math
import random
import logging
import itertools
import numpy as np

from utils.config import Config
from utils.registry_class import INFER_ENGINE

from tools import *

if __name__ == '__main__':
    cfg_update = Config(load=True)
    INFER_ENGINE.build(dict(type=cfg_update.TASK_TYPE), cfg_update=cfg_update.cfg_dict)
