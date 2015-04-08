import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce

# FILES = [f for f in filter(lambda s: s.endswith(".CSV"), os.listdir("."))]

##### transforms which can be passed to the NormalChain #####
identity = lambda x: x

def non_overlap_mean(s):
    return pd.rolling_mean(s, 10).iloc[9::10] * 10

def tail_mean(s):
    return pd.rolling_mean(s, 10).iloc[400:] * 10

def overlap_mean(s):
    return pd.rolling_mean(s, 10).iloc[9:]

#############################################################


def read_file(f):
    s = pd.read_csv(f, squeeze=True, header=None, names=['time', 'current'], index_col=['time'])
    s.index = np.round(s.index, decimals=3)
    return s


def tek_name(i):
    return "TEK000%02d.CSV" % i


##### loading good_series and bad_series dicts containing data #####

good_series = {}
for i in range(4):
    name = tek_name(i)
    s = read_file(name)
    s.index = np.round(s.index, decimals=3)
    good_series[name] = s

bad_series = {}
for i in range(10, 18):
    name = tek_name(i)
    s = read_file(name)
    s.index = np.round(s.index, decimals=3)
    bad_series[name] = s

#####################################################################


def plot_all_series(series_dict):
    """ plots a sqrt(n) x sqrt(n) subplot where n = len(series_dict) """
    dim = int(np.ceil(np.sqrt(len(series_dict))))
    subplt = plt.subplots(dim, dim)
    for ax, (name, s) in zip(subplt[1].reshape((dim*dim, )), series_dict.items()):
        s.plot(ax=ax, title = name)
    plt.show()


class SingleParamNormal(object):
    """
        Inputs:
            The single parameter Gaussian models the data mean as N(mean, sigma), by updating
            the mean and the sigma (variance of the estimated mean) but not the data_sigma
            mean - initial mean of the parameter
            sigma - initial variance of the parameter
            data_sigma - variance of the data

        Useage:
            x = SingleParamNormal(0, 1, 1)
            x.update(3)
            x.update(3.5)
            x.update(3.8)
            print(x.prob(-1))  # should be low
    """

    def __init__(self, mean, sigma, data_sigma):
        self.data_sigma = data_sigma
        self.mean = mean
        self.sigma = sigma
        
    def sample(self, num_samples=1):
        """ sample from the posterior """
        return np.random.normal(self.mean, self.data_sigma, num_samples)

    def update(self, new_point):
        """ assuming new data follows N(mean, data_sigma) distribution
            updating accoridng to Bayes with known sigma
        """
        self.mean += (new_point - self.mean) * (self.sigma**2) / (self.data_sigma**2 + self.sigma**2)
        self.sigma = np.sqrt((self.sigma**2 * self.data_sigma**2) / (self.data_sigma**2 + self.sigma**2))
        
    def sample_predictive(self, num_samples):
        """ sample from the posterior predictive """
        return np.random.normal(self.mean, np.sqrt(self.sigma**2 + self.data_sigma**2))

    def prob(self, point):
        """ what is the probability that the input point belongs? """
        p = stats.norm(self.mean, self.data_sigma)
        return p.pdf(point)


class NormalChain(object):
    """
        Inputs:
            n - number of nodes in the Gaussian chain (should be len(series) - 1)
            priors - a tuple of (mean, sigma) to initialize the parameter of each node in the chain with;
            data_sigma - the assumed variance of the input data (does not get trained in a 1-parameter model)
            tag - a name to give to the chain (not used at the moment)

        Useage:
            chain = NormalChain(999, priors=(0, 0.01), data_sigma=0.01)
            chain.transform = non_overlap_mean
            chain.update(new_series)
    """
    def __init__(self, n, priors=(0, 1), data_sigma=1, tag="good"):
        self.num_links = n
        self.nodes = [SingleParamNormal(priors[0], priors[1], data_sigma) for _ in range(n - 1)]
        self.tag = tag
        self._transform = identity
        
    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, new_transform):
        self._transform = new_transform

    def update(self, arr):
        """ arr is of length n; take diffs, which is of length n - 1 """
        diffs = np.diff(self.transform(arr))
        for node, point in zip(self.nodes, diffs):
            node.update(point)
    
    def sample_path(self):
        """ simulate a path using the learned distribution """
        new_diffs = np.array([node.sample() for node in self.nodes], dtype=np.float)
        return pd.Series(data=np.cumsum(new_diffs))
        
    def log_likelihood(self, new_data):
        """ given new_data, what is the probability that it fits here? """
        diffs = np.diff(self.transform(new_data))
        return sum(filter(lambda x: x > 1e-30, [np.log(node.prob(v)) for node, v in zip(self.nodes, diffs)]))

    def plot_path(self, num_samples=1000):
        """ simulate a bunch of paths and plot the average simulated path """
        paths = np.array([self.sample_path() for _ in range(num_samples)])
        plt.plot(paths.mean(axis=0))
        plt.show()


class Predictor(object):
    """ 
        Inputs:
            init takes named arguments, where the argument is a NormalChain and the argument
            name is some description of what the chain is modeling

        Useage: 
            p = Predictor(good=chain, bad=bad_chain)
            p.predict(s)
    """
    def __init__(self, **kwargs):
        self.chains = kwargs
        
    def predict(self, series):
        """ calculate the log-likelihood for all the chains and find the one most likely
            to correspond to the input series;

            returns the name given to the chain in kwargs
        """
        logLikes = {name: chain.log_likelihood(series) for (name, chain) in self.chains.items()}
        return reduce(lambda x, y: x if logLikes[x] > logLikes[y] else y, logLikes.keys())


def train_chain(series_dict, trans, n):
    """ convenience to create a chain from a dictionary of pd.Series objects """
    chain = NormalChain(n, priors=(0, 0.01), data_sigma = 0.01)
    chain.transform = trans
    for i, s in enumerate(series_dict.values()):
        if i > len(series_dict) - 2:
            break
        chain.update(s)
    return chain


if __name__ == "__main__":
    good_chain = train_chain(good_series, overlap_mean, 989)
    bad_chain = train_chain(bad_series, overlap_mean, 989)

    p = Predictor(good=good_chain, bad=bad_chain)
    print("good series")
    for i in range(4):
        print(p.predict(good_series[tek_name(i)]), end=" ")
    print()
    print("bad series: ", end=" ")
    for i in range(10, 18):
        print(p.predict(bad_series[tek_name(i)]), end=" ")
    print()
    # plot_all_series(bad_series)