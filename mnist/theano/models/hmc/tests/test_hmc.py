from __future__ import print_function

import numpy as np

import theano
import theano.tensor as T


def test_hmc():
    sampler = sampler_on_nd_gaussian(HMCSampler.new_from_shared_positions,
                                     burnin=1000,
                                     n_samples=1000,
                                     dim=5)

    assert abs(sampler.avg_acceptance_rate.get_value() -
               sampler.target_acceptance_rate.get_value) < .1

    assert sampler.stepsize.get_value() >= sampler.stepsize_min
    assert sampler.stepsize.get_value() <= sampler.stepsize_max
    

def sampler_on_nd_gaussian(sampler_cls, burnin, n_samples, dim=10):
    """
    """

    batchsize = 3
    rng = np.random.RandomState(123)

    # define a convariance and mu for a gaussian
    mu = np.array(rng.rand(dim) * 10, dtype=theano.config.floatX)
    cov = np.array(rng.rand(dim, dim), dtype=theano.config.floatX)
    cov = (cov + cov.T) / 2.
    cov[np.arange(dim), np.arange(dim)] = 1.0
    cov_inv = np.linalg.inv(cov)

    # define energy function for a multi-variate Gaussian
    def gaussian_energy(x):
        return 0.5 * (T.dot((x - mu), cov_inv) * (x - mu)).sum(axis=1)

    # declared shared random variable for positions
    position = rng.randn(batchsize, dim).astype(theano.config.floatX)
    position = theano.shared(position)

    # create HMC sampler
    sampler = sampler_cls(position,
                          gaussian_energy,
                          initial_stepsize=1e-3,
                          stepsize_max=0.5)

    # start with a burn-in process
    garbage = [sampler.draw() for r in range(burnin)]
    # n_samples: [n_samples, batchsize, dim]
    _samples = np.asarray([sampler.draw() for r in range(n_samples)])
    # flatten to [n_samples * batchsize, dim]
    samples = _samples.T.reshape(dim, -1).T

    print("***** Target Values *****")
    print("target mean: ", mu)
    print("target cov: ", cov)

    print("***** Empirical mean/cov using HMC *****")
    print("empirical mean: ", samples.mean(axis=0))
    print("empirical cov: ", np.cov(samples.T))

    print("***** HMC internals *****")
    print("final stepsize: ", sampler.stepsize.get_value())
    print("final acceptance rate: ", sampler.avg_acceptance_rate.get_value())

    return sampler
if __name__ == '__main__':
    test_hmc()
