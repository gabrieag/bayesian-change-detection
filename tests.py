
import math,numpy

from matplotlib import image,pyplot
from numpy import linalg,random
from scipy import interpolate

from tools import gendata,filterdata,segmentdata,plotprob,plotbound

def simpletest():

    """Simple test for the Bayesian change detection model."""

    # Set the size
    # of the problem.
    numpred=2
    numresp=3
    numpoint=200
    numseg=5

    # Set parameters for
    # generating the data.
    gainparam=0.5
    noiseparam=5.0

    # Generate a sequence of segments and, for each
    # segment, generate a set of predictor-response data.
    segbound,pred,resp=gendata(numpred,numresp,numpoint,numseg,
                               omega=gainparam*numpy.eye(numpred),
                               eta=noiseparam)

    rate=float(numseg)/float(numpoint-numseg)

    # Compute the posterior probabilities over the
    # segmentation hypotheses given the data. Then,
    # find the most likely segmentation of the sequence.
    hypotprob=filterdata(pred,resp,ratefun=rate,maxhypot=20)
    changedet=segmentdata(pred,resp,ratefun=rate,maxhypot=20)

    fig,(upperaxes,loweraxes)=pyplot.subplots(2,sharex=True)
    fig.subplots_adjust(hspace=0)

    upperaxes.set_title('Bayesian model-based change detection')
    upperaxes.set_ylabel('Response data')
    loweraxes.set_xlabel('Sequence elements')
    loweraxes.set_ylabel('Hypothesis probabilities')

    # Plot the response data.
    for i in range(numresp):
        upperaxes.plot(numpy.arange(1,numpoint+1),resp[:,i])

    # Plot the posterior probabilities
    # of the segmentation hypotheses.
    plotprob(loweraxes,hypotprob,
             cmap=pyplot.cm.gray)

    upperaxes.autoscale(False)
    loweraxes.autoscale(False)

    # Plot the changes detected by the segmentation algorithm.
    plotbound(upperaxes,changedet,
              filled=True,
              facecolor='y',
              alpha=0.2,
              edgecolor='none')
    plotbound(loweraxes,changedet,
              filled=True,
              facecolor='y',
              alpha=0.2,
              edgecolor='none')

    # Plot the true segment boundaries as vertical lines.
    plotbound(upperaxes,segbound,
              filled=False,
              color='k',
              linestyle=':')
    plotbound(loweraxes,segbound,
              filled=False,
              color='k',
              linestyle=':')

    pyplot.show()

if __name__=='__main__':
    simpletest()
