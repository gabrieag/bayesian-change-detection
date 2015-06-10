#!/usr/bin/python
"""Demonstrate Bayesian change-point detection model.

.. codeauthor:: Gabriel Agamennoni <gabriel.agamennoni@mavt.ethz.ch>
.. codeauthor:: Asher Bender <a.bender@acfr.usyd.edu.au>

"""
import os
from os import path

import numpy as np
from numpy import random
from numpy import linalg
import matplotlib.pyplot as plt

import logging

from change_detec import Bcdm
from change_detec import MatrixVariateNormalInvGamma

# Use same random data for repeatability.
np.random.seed(seed=1729)


def gen_random_data(m, n, k, l, mu=None, omega=None, sigma=None, eta=None,
                    featfun=None):
    """Randomly generate multi-variate input-output data

    First, generate a sequence of segments at random. Then, for each segment,
    generate a random coefficient matrix and a random noise covariance matrix,
    and use these to generate input-output data.
    """

    # Set default prior for the location parameter.
    if mu is None:
        mu = np.zeros([m, n])

    # Set default prior for the scale parameter.
    if omega is None:
        omega = np.eye(m)

    # Set default prior for the dispersion/noise parameter.
    if sigma is None:
        sigma = np.eye(n)

    # Set default prior for the shape parameter.
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

        # Generate the coefficient matrix and the noise covariance matrix.
        coeff, noise = MatrixVariateNormalInvGamma(mu, omega, sigma, eta).rand()
        fact = linalg.cholesky(noise).transpose()

        # Given a set of predictor data, generate a corresponding set of
        # response data.
        for j in range(k):
            y[j, :] = featfun(np.dot(x[j, :], coeff))
            y[j, :] += np.dot(random.randn(n), fact)

        X.append(x)
        Y.append(y)

    # Adjust the last element of the true boundaries for python's zero
    # indexing.
    bound[-1] -= 1

    return bound, np.concatenate(X, axis=0), np.concatenate(Y, axis=0)


def plot_probability(axes, prob, scale=None, **arg):
    """Plot hypotheses probabilities as a raster image."""

    if scale is None:
        scale = lambda x: x

    k = max(np.shape(prob)) - 1
    ind, = np.nonzero(prob.max(axis=1) > 0)
    j = ind.max()

    # Plot the posterior probabilities over segment length hypotheses.
    axes.imshow(1.0 - prob[:j+1],
                origin='lower',
                aspect='auto',
                extent=[scale(-0.5), scale(k + 0.5), -0.5, j + 0.5],
                interpolation='none',
                **arg)


def plot_segment_span(x, segments=None, **arg):
    """Plot segments as alternating vertical spans (rectangles)."""

    if segments is not None:
        for i in range(len(segments) - 1):
            if i % 2 == 0:
                plt.axvspan(x[segments[i]], x[segments[i + 1]], **arg)

    else:
        for i in range(len(x) - 1):
            if i % 2 == 0:
                plt.axvspan(x[i], x[i + 1], **arg)


def plot_segment_boundaries(x, segments=None, **args):
    """Plot segment boundaries as vertical lines."""

    if segments is not None:
        for i in range(len(segments)):
            plt.axvline(x[segments[i]], **args)

    else:
        for i in range(len(x)):
            plt.axvline(x[i], **args)


def plot_segment_models(x, segments, basisfun=None, **args):
    pass


def random_data():
    """Simple test with synthetic data."""

    # Set the size of the problem.
    numpred = 2
    numresp = 3
    numpoint = 200
    numseg = 5

    # Set parameters for generating the data.
    coeffparam = 0.5
    noiseparam = 5.0

    # Generate a sequence of segments and, for each segment, generate a set of
    # predictor-response data.
    segbound, X, Y = gen_random_data(numpred, numresp, numpoint, numseg,
                                     omega=coeffparam*np.eye(numpred),
                                     eta=noiseparam)

    rate = float(numseg) / float(numpoint - numseg)

    # Compute the posterior probabilities over segment length hypotheses. Then,
    # find the most likely segmentation of the sequence.
    bcdm_probabilities = Bcdm(alg='sumprod', ratefun=rate)
    bcdm_segments = Bcdm(alg='maxprod', ratefun=rate)

    # Update the segment length hypotheses given the data.
    for x, y in zip(X, Y):
        bcdm_probabilities.update(x, y)
        bcdm_segments.update(x, y)

    # Recover the hypothesis probabilities and back-trace to find the most
    # likely segmentation of the sequence.
    hypotheses_probability = bcdm_probabilities.infer()
    segments = bcdm_segments.infer()

    # Create subplots with shared X-axis.
    fig, (upperaxes, loweraxes) = plt.subplots(2, sharex=True)
    fig.subplots_adjust(hspace=0)

    # Plot the response data.
    t = np.arange(1, numpoint + 1)
    for i in range(numresp):
        upperaxes.plot(t, Y[:, i])

    # Plot the posterior probabilities over segment length hypotheses.
    plot_probability(loweraxes, hypotheses_probability, cmap=plt.cm.gray)

    # Plot the changes detected by the segmentation algorithm as alternating
    # coloured spans. Plot the true segment boundaries as vertical lines.
    for ax in (upperaxes, loweraxes):
        plt.sca(ax)
        plot_segment_span(t, segments, facecolor='y', alpha=0.2, edgecolor='none')
        plot_segment_boundaries(t, segbound, color='k', linestyle=':')
        ax.set_xlim([0, numpoint])

    fig.canvas.set_window_title('Random data')
    upperaxes.set_title('Random data')
    upperaxes.set_ylabel('Response')
    loweraxes.set_xlabel('Sequence number')
    loweraxes.set_ylabel('Hypothesis probability')


def non_sinusoidal():
    """Simple test with triangular wave data."""

    rate = 0.001
    omega = 1.0e-3 * np.eye(2)
    sigma = 1.0e-6 * np.eye(3)
    samples = 1000
    basis = lambda x: np.array([[1.0, x]])

    # create non-sinusoidal waveform functions.
    square_wave = lambda x: np.sign(np.sin(x))
    sawtooth_wave = lambda a, x: 2 * ((x/a) - np.floor(0.5 + (x/a)))
    triangle_wave = lambda a, x: 2 * np.abs(sawtooth_wave(a, x)) - 1

    # Create input and outputs.
    X = np.linspace(0, 3*2*np.pi, samples).reshape(samples, 1)
    Y = np.hstack([square_wave(X),
                   triangle_wave(2*np.pi, X - np.pi/2),
                   sawtooth_wave(2*np.pi, X + np.pi/3)])

    # Create Gaussian noise.
    Y += np.vstack([0.025 * np.random.randn(samples),
                    0.1 * np.random.randn(samples),
                    0.05 * np.random.randn(samples)]).T

    # Determine location of true boundaries.
    true_boundaries = np.hstack((np.pi * np.arange(0, 7),
                                 np.pi * np.arange(0, 6) + np.pi/2,
                                 2*np.pi * np.arange(0, 4) + np.pi - np.pi/3))

    true_boundaries = np.sort(true_boundaries[true_boundaries <= max(X)])

    # Compute the posterior probabilities over segment length hypotheses. Then,
    # find the most likely segmentation of the sequence.
    bcdm_probabilities = Bcdm(alg='sumprod',
                              ratefun=rate,
                              basisfunc=basis,
                              omega=omega,
                              sigma=sigma)

    bcdm_segments = Bcdm(alg='maxprod',
                         ratefun=rate,
                         basisfunc=basis,
                         omega=omega,
                         sigma=sigma)

    # Update the segment length hypotheses given the data.
    for x, y in zip(X, Y):
        y = np.array([y])
        basis_t = lambda xt: basis(xt - x)
        bcdm_probabilities.update(x, y, basisfunc=basis_t)
        bcdm_segments.update(x, y, basisfunc=basis_t)

    # Recover the hypothesis probabilities and back-trace to find the most
    # likely segmentation of the sequence.
    hypotheses_probability = bcdm_probabilities.infer()
    segments = bcdm_segments.infer()

    # Create subplots with shared X-axis.
    fig, (upperaxes, loweraxes) = plt.subplots(2, sharex=False)

    # Plot the response data.
    for i in range(Y.shape[1]):
        upperaxes.plot(X, Y[:, i])

    # Plot the posterior probabilities over segment length hypotheses.
    plot_probability(loweraxes, hypotheses_probability, cmap=plt.cm.gray)

    # Plot the changes detected by the segmentation algorithm as alternating
    # coloured spans. Plot the true segment boundaries as vertical lines.
    plt.sca(upperaxes)
    plot_segment_span(X, segments, facecolor='y', alpha=0.2, edgecolor='none')
    plot_segment_boundaries(true_boundaries, color='k', linestyle=':')

    plt.sca(loweraxes)
    plot_segment_span(segments, facecolor='y', alpha=0.2, edgecolor='none')
    plot_segment_boundaries(samples * true_boundaries / max(X),
                            color='k', linestyle=':')

    upperaxes.set_xlim([0, max(X)])
    loweraxes.set_xlim([0, len(X)])

    fig.canvas.set_window_title('Non-sinusoidal data')
    upperaxes.set_title('Non-sinusoidal data')
    upperaxes.set_ylabel('Response')
    upperaxes.set_xlabel('Predictor')
    loweraxes.set_xlabel('Sequence number')
    loweraxes.set_ylabel('Hypothesis probability')


def well_data():
    """Simple test with nuclear response data collected the drilling of a well

    Segment the well log data used in Fearnhead and Clifford (1996). This data
    consist of measurements of the nuclear magnetic response of underground
    rocks, collected during the drilling of a well bore. The data are composed
    of piecewise constant segments, each segment relating to a stratum with a
    single type of rock. The jump discontinuities between segments occur at the
    boundaries between rock strata.

    P. Fearnhead and P. Clifford, "Online Inference for Hidden Markov Models
    via Particle Filters," Journal of the Royal Statistical Society: Series B
    (Statistical Methodology), Vol. 65, Issue 4, pp. 887-889, November 2003.
    """

    loc = 1.0e5
    scale = 1.0e4
    rate = 1.0e-2

    val = []

    # Store the absolute path to the file containing the data.
    abspath = path.realpath(path.join(os.getcwd(), 'data'))
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

    # Compute the posterior probabilities over segment length hypotheses. Then,
    # find the most likely sequence segmentation.
    bcdm_probabilities = Bcdm(alg='sumprod', **kwargs)
    bcdm_segments = Bcdm(alg='maxprod', **kwargs)

    # Update the segment length hypotheses given the data.
    for x, y in zip(X, Y):
        bcdm_probabilities.update(x, y)
        bcdm_segments.update(x, y)

    # Recover the hypothesis probabilities and back-trace to find the most
    # likely segmentation of the sequence.
    hypotheses_probability = bcdm_probabilities.infer()
    segments = bcdm_segments.infer()

    # Create subplots with shared X-axis.
    fig, (upperaxes, loweraxes) = plt.subplots(2, sharex=True)
    fig.subplots_adjust(hspace=0)

    # Plot the response data.
    t = np.arange(1, len(val) + 1)
    upperaxes.plot(t, Y[:])

    # Plot the posterior probabilities over segment length hypotheses.
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

    # Create console logger.
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    logger.info('Running random data test ...')
    random_data()

    logger.info('Running triangular wave data test ...')
    non_sinusoidal()

    logger.info('Running well log data test ...')
    well_data()

    plt.show()
