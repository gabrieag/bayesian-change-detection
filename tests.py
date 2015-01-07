#!/usr/bin/python
"""Demonstrate Bayesian change-point detection model.

.. codeauthor:: Gabriel Agamennoni <abriel.agamennoni@mavt.ethz.ch>
.. codeauthor:: Asher Bender <a.bender@acfr.usyd.edu.au>

"""
import os
from os import path

import numpy as np
from numpy import random
from numpy import linalg
import matplotlib.pyplot as plt

from segmentation import Bcdm
from segmentation import MatrixVariateNormalInvGamma


def random_segments(m, n, k, l, mu=None, omega=None, sigma=None, eta=None,
                    featfun=None):
    """Generate random segmented multi-variate linear model data."""

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

    # Adjust the last element of the true boundaries for python's zero
    # indexing.
    bound[-1] -= 1

    return bound, np.concatenate(X, axis=0), np.concatenate(Y, axis=0)


def plot_probability(axes, prob, scale=None, **arg):

    """Plot hypotheses probability as a raster."""

    if scale is None:
        scale = lambda x: x

    k = max(np.shape(prob)) - 1
    ind, = np.nonzero(prob.max(axis=1) > 0)
    j = ind.max()

    # Plot the posterior probabilities of the segmentation hypotheses.
    axes.imshow(1.0 - prob[:j+1],
                origin='lower',
                aspect='auto',
                extent=[scale(-0.5), scale(k + 0.5), -0.5, j + 0.5],
                interpolation='none',
                **arg)


def plot_segment_span(x, segments, **arg):
    """Plot segments as alternating vertical spans (rectangles)."""

    for i in range(len(segments) - 1):
        if i % 2 == 0:
            plt.axvspan(x[segments[i]], x[segments[i + 1]], **arg)


def plot_segment_boundaries(x, segments, **args):
    """Plot segment boundaries as vertical lines."""

    for i in range(len(segments)):
        plt.axvline(x[segments[i]], **args)


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
    hypotheses_probability = bcdm_probabilities.segment()
    segments = bcdm_segments.segment()

    # Create subplots with shared X-axis.
    fig, (upperaxes, loweraxes) = plt.subplots(2, sharex=True)
    fig.subplots_adjust(hspace=0)

    # Plot the response data.
    t = np.arange(1, numpoint + 1)
    for i in range(numresp):
        upperaxes.plot(t, Y[:, i])

    # Plot the posterior probabilities of the segmentation hypotheses.
    plot_probability(loweraxes, hypotheses_probability, cmap=plt.cm.gray)

    # Plot the changes detected by the segmentation algorithm as alternating
    # coloured spans. Plot the true segment boundaries as vertical lines.
    for ax in (upperaxes, loweraxes):
        plt.sca(ax)
        plot_segment_span(t, segments, facecolor='y', alpha=0.2, edgecolor='none')
        plot_segment_boundaries(t, segbound, color='k', linestyle=':')
        ax.set_xlim([0, numpoint])

    fig.canvas.set_window_title('Synthetic data')
    upperaxes.set_title('Synthetic data')
    upperaxes.set_ylabel('Response')
    loweraxes.set_xlabel('Sequence number')
    loweraxes.set_ylabel('Hypothesis probability')


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
    hypotheses_probability = bcdm_probabilities.segment()
    segments = bcdm_segments.segment()

    # Create subplots with shared X-axis.
    fig, (upperaxes, loweraxes) = plt.subplots(2, sharex=True)
    fig.subplots_adjust(hspace=0)

    # Plot the response data.
    t = np.arange(1, len(val) + 1)
    upperaxes.plot(t, Y[:])

    # Plot the posterior probabilities of the segmentation hypotheses.
    plot_probability(loweraxes, hypotheses_probability, cmap=plt.cm.gray)

    # Plot the changes detected by the segmentation algorithm as alternating
    # coloured spans. Plot the true segment boundaries as vertical lines.
    for ax in (upperaxes, loweraxes):
        plt.sca(ax)
        plot_segment_span(t, segments, facecolor='y', alpha=0.2, edgecolor='none')
        ax.set_xlim([0, len(val)])

    fig.canvas.set_window_title('Drilling data')
    upperaxes.set_title('Drilling data')
    upperaxes.set_ylabel('Nuclear response')
    loweraxes.set_xlabel('Measurement number')
    loweraxes.set_ylabel('Hypothesis probability')


if __name__ == '__main__':
    synthetic_data()
    well_data()
    plt.show()
