
from cec2013.cec2013 import *
import numpy as np
import random

for i in range(1,21):
    total = [0,0,0,0]
    for j in range(1,51):
        opts_log_filename = "logs\\problem%03drun%03d_opts_log.txt" % (i, j)
        opts_log_file = open(opts_log_filename, "r")
