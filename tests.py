import os
import numpy
from os import path
from matplotlib import pyplot
from tools import gendata
from tools import filterdata
from tools import segmentdata
from tools import plotprob
from tools import plotbound


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
    segbound, pred, resp = gendata(numpred, numresp, numpoint, numseg,
                                   omega=gainparam*numpy.eye(numpred),
                                   eta=noiseparam)

    rate = float(numseg) / float(numpoint - numseg)

    # Compute the posterior probabilities of the segmentation hypotheses. Then,
    # find the most likely segmentation of the sequence.
    hypotprob = filterdata(pred, resp, ratefun=rate)
    changedet = segmentdata(pred, resp, ratefun=rate)

    fig, (upperaxes, loweraxes) = pyplot.subplots(2, sharex=True)
    fig.subplots_adjust(hspace=0)

    upperaxes.set_title('Synthetic data')
    upperaxes.set_ylabel('Response')
    loweraxes.set_xlabel('Sequence number')
    loweraxes.set_ylabel('Hypothesis probability')

    # Plot the response data.
    for i in range(numresp):
        upperaxes.plot(numpy.arange(1, numpoint + 1), resp[:, i])

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

    val=[]

    # Store the absolute path to the file containing the data.
    abspath = path.realpath(path.join(os.getcwd(), path.dirname(__file__)))
    abspath = path.join(abspath,'well-data.txt')

    # Read the data.
    with open(abspath, 'r') as file:
        for line in file:
            try:
                val.append(float(line))
            except:
                pass

    # Format the data.
    pred = numpy.ones([len(val), 1])
    resp = numpy.array(val).reshape([len(val), 1])

    loc = numpy.array([(loc, )])
    scale = numpy.array([(scale, )])

    # Compute the posterior probabilities of the segmentation hypotheses. Then,
    # find the most likely sequence segmentation.
    hypotprob = filterdata(pred, resp, mu=loc, sigma=scale, ratefun=rate)
    changedet = segmentdata(pred, resp, mu=loc, sigma=scale, ratefun=rate)

    fig, (upperaxes, loweraxes) = pyplot.subplots(2, sharex=True)
    fig.subplots_adjust(hspace=0)

    upperaxes.set_title('Drilling data')
    upperaxes.set_ylabel('Nuclear response')
    loweraxes.set_xlabel('Measurement number')
    loweraxes.set_ylabel('Hypothesis probability')

    # Plot the data.
    upperaxes.plot(numpy.arange(1, len(val) + 1), resp[:])

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


if __name__=='__main__':
    synthdata()
    welldata()
