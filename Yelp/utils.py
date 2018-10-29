import math
import pickle


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def logistic_anneal(step, a=0.0025, x0=2500):
    return 1. / (1. + math.exp(-a * (step - x0)))

def linear_anneal(step, x0, initial=0.01):
    return min(1., initial + step / x0)
