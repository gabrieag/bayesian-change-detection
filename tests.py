#!/usr/bin/python
import os
from os import path

import numpy as np
from numpy import random
from numpy import linalg
from matplotlib import pyplot

from infeng import Bcdm
from infeng import MatrixVariateNormalInvGamma


def random_segments(m, n, k, l, mu=None, omega=None, sigma=None, eta=None,
                    featfun=None):

    # Set uninformative prior for the location parameter.
    if mu is None:
        mu = np.zeros([m, n])

    # Set uninformative prior for the scale parameter.
    if omega is None:
        omega = np.eye(m)

    # Set uninformative prior for the dispersion/noise parameter.
    if sigma is None:
        sigma = np.eye(n)

    # Set uninformative prior for the shape parameter.
    if eta is None:
        eta = n

    featfun = featfun if callable(featfun) else lambda x: x

    # Generate the segment boundaries.
    bound = random.permutation(np.arange(k - 1) + 1)
    bound = np.concatenate([np.array([0]),
                            np.sort(bound[:l-1]),
                            np.array([k])])

    # For each segment generate a set of predictor-response data.
    X, Y = list(), list()
    for i in range(len(bound) - 1):

        # Generate random predictor (input) data and pre-allocate memory for
        # response (output) data.
        k = bound[i + 1] - bound[i]
        x = random.rand(k, m)
        y = np.zeros((k, n))

        # Generate the gain and noise parameters.
        gain, noise = MatrixVariateNormalInvGamma(mu, omega, sigma, eta).rand()
        fact = linalg.cholesky(noise).transpose()

        # Given a set of predictor data, generate a corresponding set of
        # response data.
        for j in range(k):
            y[j, :] = featfun(np.dot(x[j, :], gain))
            y[j, :] += np.dot(random.randn(n), fact)

        X.append(x)
        Y.append(y)

    return bound, np.concatenate(X, axis=0), np.concatenate(Y, axis=0)


def plotprob(axes, prob, scale=None, **arg):

    k = max(np.shape(prob)) - 1

    if scale is None:
        scale = lambda x: x

    ind, = np.nonzero(prob.max(axis=1) > 0)
    j = ind.max()

    # Plot the posterior probabilities of the segmentation hypotheses.
    axes.imshow(1.0-prob[:j+1],
                origin='lower',
                aspect='auto',
                extent=[scale(-0.5), scale(k + 0.5), -0.5, j + 0.5],
                interpolation='none',
                **arg)


def plotbound(axes, bound, scale=None, filled=False, **arg):

    lower, upper = axes.get_ylim()

    if scale is None:
        scale = lambda x: x

    if filled:
        x, y = [], []

        # Store the coordinates of the boundary mid-points.
        for i in range(0, len(bound) - 1, 2):
            x += [scale(bound[i] + 0.5),
                  scale(bound[i + 1] + 0.5),
                  scale(bound[i + 1] + 0.5),
                  scale(bound[i] + 0.5),
                  np.nan]
            y += [lower, lower, upper, upper, np.nan]

        x.pop()
        y.pop()

        # Plot the segment boundaries as filled rectangles.
        axes.fill(x, y, **arg)

    else:
        x, y = [], []

        # Store the coordinates of the boundary mid-points.
        for i in range(1, len(bound) - 1):
            x += [scale(bound[i] + 0.5),
                  scale(bound[i] + 0.5),
                  np.nan]

            y += [lower, upper, np.nan]

        x.pop()
        y.pop()

        # Plot the segment boundaries as vertical lines.
        axes.plot(x, y, **arg)


def synthetic_data():
    """Simple test with synthetic data."""

    # Set the size of the problem.
    numpred = 2
    numresp = 3
    numpoint = 200
    numseg = 5

    # Set parameters for generating the data.
    gainparam = 0.5
    noiseparam = 5.0

    # Generate a sequence of segments and, for each segment, generate a set of
    # predictor-response data.
    segbound, X, Y = random_segments(numpred, numresp, numpoint, numseg,
                                     omega=gainparam*np.eye(numpred),
                                     eta=noiseparam)

    rate = float(numseg) / float(numpoint - numseg)

    # Compute the posterior probabilities of the segmentation hypotheses. Then,
    # find the most likely segmentation of the sequence.
    bcdm_probabilities = Bcdm(alg='sumprod', ratefun=rate)
    bcdm_segments = Bcdm(alg='maxprod', ratefun=rate)

    # Update the segmentation hypotheses given the data.
    bcdm_probabilities.block_update(X, Y)
    bcdm_segments.block_update(X, Y)

    # Recover the hypothesis probabilities and back-trace to find the most
    # likely segmentation of the sequence.
    hypotprob = bcdm_probabilities.segment()
    changedet = bcdm_segments.segment()

    fig, (upperaxes, loweraxes) = pyplot.subplots(2, sharex=True)
    fig.subplots_adjust(hspace=0)

    upperaxes.set_title('Synthetic data')
    upperaxes.set_ylabel('Response')
    loweraxes.set_xlabel('Sequence number')
    loweraxes.set_ylabel('Hypothesis probability')

    # Plot the response data.
    for i in range(numresp):
        upperaxes.plot(np.arange(1, numpoint + 1), Y[:, i])

    # Plot the posterior probabilities of the segmentation hypotheses.
    plotprob(loweraxes, hypotprob, cmap=pyplot.cm.gray)

    upperaxes.autoscale(False)
    loweraxes.autoscale(False)

    # Plot the changes detected by
    # the segmentation algorithm.
    plotbound(upperaxes,
              changedet,
              filled=True,
              facecolor='y',
              alpha=0.2,
              edgecolor='none')

    plotbound(loweraxes,
              changedet,
              filled=True,
              facecolor='y',
              alpha=0.2,
              edgecolor='none')

    # Plot the true segment boundaries as vertical lines.
    plotbound(upperaxes,
              segbound,
              filled=False,
              color='k',
              linestyle=':')

    plotbound(loweraxes,
              segbound,
              filled=False,
              color='k',
              linestyle=':')

    fig.canvas.set_window_title('Synthetic data')

    pyplot.show()


def well_data():
    """Nuclear response data collected during the drilling of a well."""

    loc = 1.0e5
    scale = 1.0e4
    rate = 1.0e-2

    val = []

    # Store the absolute path to the file containing the data.
    abspath = path.realpath(path.join(os.getcwd(), path.dirname(__file__)))
    abspath = path.join(abspath, 'well-data.txt')

    # Read the data.
    with open(abspath, 'r') as file:
        for line in file:
            try:
                val.append(float(line))
            except:
                pass

    # Format the data.
    X = np.ones([len(val), 1])
    Y = np.array(val).reshape([len(val), 1])

    loc = np.array([(loc, )])
    scale = np.array([(scale, )])

    kwargs = {'ratefun': rate,
              'mu': loc,
              'sigma': scale}

    # Compute the posterior probabilities of the segmentation hypotheses. Then,
    # find the most likely sequence segmentation.
    bcdm_probabilities = Bcdm(alg='sumprod', **kwargs)
    bcdm_segments = Bcdm(alg='maxprod', **kwargs)

    # Update the segmentation hypotheses given the data, one point at a time.
    for i in range(X.shape[0]):
        bcdm_probabilities.update(X[i, :], Y[i, :])
        bcdm_segments.update(X[i, :], Y[i, :])

    # Recover the hypothesis probabilities and back-trace to find the most
    # likely segmentation of the sequence.
    hypotprob = bcdm_probabilities.segment()
    changedet = bcdm_segments.segment()

    fig, (upperaxes, loweraxes) = pyplot.subplots(2, sharex=True)
    fig.subplots_adjust(hspace=0)

    upperaxes.set_title('Drilling data')
    upperaxes.set_ylabel('Nuclear response')
    loweraxes.set_xlabel('Measurement number')
    loweraxes.set_ylabel('Hypothesis probability')

    # Plot the data.
    upperaxes.plot(np.arange(1, len(val) + 1), Y[:])

    # Plot the posterior probabilities of the segmentation hypotheses.
    plotprob(loweraxes, hypotprob, cmap=pyplot.cm.gray)

    upperaxes.autoscale(False)
    loweraxes.autoscale(False)

    # Plot the changes detected by the segmentation algorithm.
    plotbound(upperaxes,
              changedet,
              filled=True,
              facecolor='y',
              alpha=0.2,
              edgecolor='none')

    plotbound(loweraxes,
              changedet,
              filled=True,
              facecolor='y',
              alpha=0.2,
              edgecolor='none')

    fig.canvas.set_window_title('Drilling data')

    pyplot.show()


if __name__ == '__main__':
    synthetic_data()
    well_data()
