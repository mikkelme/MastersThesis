import sys
sys.path.append('../') # parent folder: MastersThesis


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
# from torchvision import models, transforms

import os
# import PIL.Image
# import pandas as pd
import csv

# from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# from sklearn.metrics import average_precision_score
import numpy as np
import matplotlib.pyplot as plt

import random


from torchmetrics.functional import r2_score


# --- Plotting --- #
from plot_set import *