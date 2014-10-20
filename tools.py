
import numpy

from numpy import random

# Import the inference engine.
from infeng import engine

def gendata(m,n,k,l,mu=None,omega=None,sigma=None,eta=None,**arg):

    # Create an inference engine
    # of the appropriate size.
    eng=engine(m,n)

    # Set the hyper-parameters of the model.
    eng.setparam(mu=mu,omega=omega,sigma=sigma,eta=eta)

    # Generate the segment boundaries.
    bound=random.permutation(numpy.arange(k-1)+1)
    bound=numpy.concatenate([numpy.array([0]),
                             numpy.sort(bound[:l-1]),
                             numpy.array([k])])

    pred,resp=[],[]

    # For each segment generate a set of predictor-response data.
    for k,(i,j) in enumerate(zip(bound[:-1],bound[1:])):
        pred.append(random.rand(j-i,m))
        resp.append(numpy.concatenate(eng.sim(
            *numpy.split(pred[k],j-i,axis=0)),axis=0))

    return bound,numpy.concatenate(pred,axis=0),numpy.concatenate(resp,axis=0)

def filterdata(pred,resp,mu=None,omega=None,sigma=None,eta=None,**arg):

    k,m=numpy.shape(pred)
    k,n=numpy.shape(resp)

    # Create an inference engine
    # of the appropriate size to run
    # the sum-product algorithm.
    eng=engine(m,n,alg='sumprod')

    # Set the hyper-parameters of the model.
    eng.setparam(mu=mu,omega=omega,sigma=sigma,eta=eta)

    # Allocate space for storing
    # the posterior probabilities of
    # the segmentation hypotheses.
    prob=numpy.zeros([k+1,k+1])

    # Initialize the engine.
    eng.init()

    # Initialize the probabilities.
    for j,alpha in eng.state():
        prob[j,0]=alpha

    for i in range(k):

        # Update the segmentation hypotheses
        # given the data, one point at a time.
        eng.update(pred[i,:],resp[i,:],**arg)

        # Update the probabilities.
        for j,alpha in eng.state():
            prob[j,i+1]=alpha

    return prob

def segmentdata(pred,resp,mu=None,omega=None,sigma=None,eta=None,**arg):

    k,m=numpy.shape(pred)
    k,n=numpy.shape(resp)

    # Create an inference engine
    # of the appropriate size to run
    # the max-product algorithm.
    eng=engine(m,n,alg='maxprod')

    # Set the hyper-parameters of the model.
    eng.setparam(mu=mu,omega=omega,sigma=sigma,eta=eta)

    # Initialize the engine.
    eng.init()

    for i in range(k):

        # Update the segmentation hypotheses
        # given the data, one point at a time.
        eng.update(pred[i,:],resp[i,:],**arg)

    # Backtrack to find the most likely
    # segmentation of the sequence.
    return eng.segment()

def plotprob(axes,prob,scale=None,**arg):

    k=max(numpy.shape(prob))-1

    if scale is None:
        scale=lambda x:x

    ind,=numpy.nonzero(prob.max(axis=1)>0)
    j=ind.max()

    # Plot the posterior probabilities
    # of the segmentation hypotheses.
    axes.imshow(1.0-prob[:j+1],
                origin='lower',
                aspect='auto',
                extent=[scale(-0.5),scale(k+0.5),-0.5,j+0.5],
                interpolation='none',
                **arg)

def plotbound(axes,bound,scale=None,filled=False,**arg):

    lower,upper=axes.get_ylim()

    if scale is None:
        scale=lambda x:x

    if filled:

        x,y=[],[]
    
        # Store the coordinates of the boundary mid-points.
        for i in range(0,len(bound)-1,2):
            x+=[scale(bound[i]+0.5),
                scale(bound[i+1]+0.5),
                scale(bound[i+1]+0.5),
                scale(bound[i]+0.5),
                numpy.nan]
            y+=[lower,lower,upper,upper,numpy.nan]

        x.pop()
        y.pop()

        # Plot the segment boundaries
        # as filled rectangles.
        axes.fill(x,y,**arg)

    else:

        x,y=[],[]

        # Store the coordinates
        # of the boundary mid-points.
        for i in range(1,len(bound)-1):
            x+=[scale(bound[i]+0.5),
                scale(bound[i]+0.5),
                numpy.nan]
            y+=[lower,upper,numpy.nan]

        x.pop()
        y.pop()

        # Plot the segment boundaries
        # as vertical lines.
        axes.plot(x,y,**arg)
