import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king/krkopt.data'

#Original dataset has columns (a,1), (b,3) and (c,2) which provides coordinates of the White King, White Rook, and Black King, respectively
#Letter columns provide the column position and number columns provide the row positions
dataset = pd.read_csv(dataset_url)

