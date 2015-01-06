#!/usr/bin/python
import os
import numpy
from os import path
from matplotlib import pyplot

from numpy import random
from infeng import Bcdm
from infeng import filterdata
from infeng import segmentdata


def gendata(m, n, k, l, mu=None, omega=None, sigma=None, eta=None, **arg):

    # Create an inference engine of the appropriate size.
    bcdm = Bcdm(mu=mu, omega=omega, sigma=sigma, eta=eta)

    # Generate the segment boundaries.
    bound = random.permutation(numpy.arange(k - 1) + 1)
    bound = numpy.concatenate([numpy.array([0]),
                               numpy.sort(bound[:l-1]),
                               numpy.array([k])])

    # For each segment generate a set of predictor-response data.
    X, Y = [], []
    for k, (i, j) in enumerate(zip(bound[:-1], bound[1:])):
        X.append(random.rand(j - i, m))
        Y.append(numpy.concatenate(bcdm.sim(numpy.split(X[k], j - i, axis=0), n),
                                   axis=0))

    return bound, numpy.concatenate(X, axis=0), numpy.concatenate(Y, axis=0)


def plotprob(axes, prob, scale=None, **arg):

    k = max(numpy.shape(prob)) - 1

    if scale is None:
        scale = lambda x: x

    ind, = numpy.nonzero(prob.max(axis=1) > 0)
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
                  numpy.nan]
            y += [lower, lower, upper, upper, numpy.nan]

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
                  numpy.nan]

            y += [lower, upper, numpy.nan]

        x.pop()
        y.pop()

        # Plot the segment boundaries as vertical lines.
        axes.plot(x, y, **arg)


def synthdata():
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
    segbound, X, Y = gendata(numpred, numresp, numpoint, numseg,
                             omega=gainparam*numpy.eye(numpred),
                             eta=noiseparam)

    rate = float(numseg) / float(numpoint - numseg)

    # Compute the posterior probabilities of the segmentation hypotheses. Then,
    # find the most likely segmentation of the sequence.
    hypotprob = filterdata(X, Y, ratefun=rate)
    changedet = segmentdata(X, Y, ratefun=rate)

    fig, (upperaxes, loweraxes) = pyplot.subplots(2, sharex=True)
    fig.subplots_adjust(hspace=0)

    upperaxes.set_title('Synthetic data')
    upperaxes.set_ylabel('Response')
    loweraxes.set_xlabel('Sequence number')
    loweraxes.set_ylabel('Hypothesis probability')

    # Plot the response data.
    for i in range(numresp):
        upperaxes.plot(numpy.arange(1, numpoint + 1), Y[:, i])

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


def welldata():
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
    X = numpy.ones([len(val), 1])
    Y = numpy.array(val).reshape([len(val), 1])

    loc = numpy.array([(loc, )])
    scale = numpy.array([(scale, )])

    # Compute the posterior probabilities of the segmentation hypotheses. Then,
    # find the most likely sequence segmentation.
    hypotprob = filterdata(X, Y, mu=loc, sigma=scale, ratefun=rate)
    changedet = segmentdata(X, Y, mu=loc, sigma=scale, ratefun=rate)

    fig, (upperaxes, loweraxes) = pyplot.subplots(2, sharex=True)
    fig.subplots_adjust(hspace=0)

    upperaxes.set_title('Drilling data')
    upperaxes.set_ylabel('Nuclear response')
    loweraxes.set_xlabel('Measurement number')
    loweraxes.set_ylabel('Hypothesis probability')

    # Plot the data.
    upperaxes.plot(numpy.arange(1, len(val) + 1), Y[:])

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
    synthdata()
    welldata()
