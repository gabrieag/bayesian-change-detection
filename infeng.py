
import math,numpy

from numpy import linalg,random

# Import the module-specific class.
from __util__ import suffstat

def accum(x,y):
    return max(x,y)+math.log1p(math.exp(-abs(x-y)))

class struct():
    def __init__(self,**arg):
        for key,val in arg.items():
            self.__dict__[key]=val

class engine():

    """Inference engine for the Bayesian change detection model."""

    def __init__(self,m,n,alg='sumprod'):

        # The number of predictors and responses
        # must be both positive integer scalars.
        assert m>0 and n>0

        # The inference algorithm must be
        # either sum-product or max-product.
        assert alg in ['sumprod','maxprod']

        self.__size__=m,n
        self.__alg__=alg.lower()

        # Set default values
        # for the parameters.
        mu=numpy.zeros([m,n])
        omega=numpy.eye(m)
        sigma=numpy.eye(n)
        eta=n

        self.__param__=mu,omega,sigma,eta

        self.__hypot__=[]
        self.__ind__=None

    def getparam(self):
        return self.__param__

    def setparam(self,mu=None,omega=None,sigma=None,eta=None):

        m,n=self.__size__

        if mu is None:
            mu=self.__param__[0]
        else:

            # Check that the location parameter
            # is a matrix of finite numbers.
            assert numpy.ndim(mu)==2 and\
                   numpy.shape(mu)==(m,n) and\
                   not numpy.isnan(mu).any() and\
                   numpy.isfinite(mu).all()

        if omega is None:
            omega=self.__param__[1]
        else:

            # Check that the scale parameter is a
            # symmetric, positive-definite matrix.
            assert numpy.ndim(omega)==2 and\
                   numpy.shape(omega)==(m,m) and\
                   not numpy.isnan(omega).any() and\
                   numpy.isfinite(omega).all() and\
                   numpy.allclose(numpy.transpose(omega),omega) and\
                   linalg.det(omega)>0.0

        if sigma is None:
            sigma=self.__param__[2]
        else:

            # Check that the dispersion parameter is
            # a symmetric, positive-definite matrix.
            assert numpy.ndim(sigma)==2 and\
                   numpy.shape(sigma)==(n,n) and\
                   not numpy.isnan(sigma).any() and\
                   numpy.isfinite(sigma).all() and\
                   numpy.allclose(numpy.transpose(sigma),sigma) and\
                   linalg.det(sigma)>0.0

        if eta is None:
            eta=self.__param__[3]
        else:

            # Check that the shape parameter is
            # a number greater than one minus the
            # number of degrees of freedom.
            assert numpy.isscalar(eta) and\
                   not numpy.isnan(eta) and\
                   numpy.isfinite(eta) and eta>n-1.0

        self.__param__=mu,omega,sigma,eta

    def init(self,featfun=None):

        stat=suffstat(*self.__param__)

        fun=featfun if callable(featfun) else lambda x:x

        # Create the initial hypothesis, which states
        # that the first segment is about to begin.
        self.__hypot__=[struct(count=0,
                               logprob=0.0,
                               stat=stat,
                               logconst=stat.logconst(),
                               featfun=fun)]

        if self.__alg__=='maxprod':

            # The max-product algorithm
            # involves keeping track of the
            # most likely hypotheses.
            self.__ind__=[]

    def sim(self,*pred,featfun=None):

        m,n=self.__size__

        fun=featfun if callable(featfun) else lambda x:x

        # Generate the gain and noise parameters.
        gain,noise=suffstat(*self.__param__).rand()

        resp=[]

        fact=linalg.cholesky(noise).transpose()

        # Given a set of predictor
        # data, aenerate a corresponding
        # set of response data.
        for pred in pred:
            if numpy.ndim(pred)>1:
                k,m=numpy.shape(pred)
                shape=[k,n]
            else:
                shape=[n]
            resp.append(fun(numpy.dot(pred,gain))
                        +numpy.dot(random.randn(*shape),fact))

        return resp

    def update(self,pred,resp,featfun=None,ratefun=0.1):

        m,n=self.__size__

        # Deduce the number of points.
        if numpy.ndim(pred)>1:
            k,_=numpy.shape(pred)
        else:
            k=numpy.size(pred)

        fun=featfun if callable(featfun) else lambda x:x

        if not callable(ratefun):
            rate=float(ratefun)
            ratefun=lambda x:rate

        loglik=-numpy.inf
        logmax=-numpy.inf
        logsum=-numpy.inf

        ind=numpy.nan

        for i,hypot in enumerate(self.__hypot__):

            # Update the sufficient statistics.
            hypot.stat.update(hypot.featfun(pred),resp)

            # Compute the log-normalization constant
            # of the posterior parameter distribution.
            logconst=hypot.logconst
            hypot.logconst=hypot.stat.logconst()

            # Evaluate the log-density of the predictive distribution.
            logdens=hypot.logconst-logconst-k*(0.5*m*n)*math.log(2.0*math.pi)

            # Increment the counter.
            hypot.count+=1

            aux=math.log(ratefun(hypot.count))+logdens+hypot.logprob

            # Accumulate the log-likelihood of the data.
            loglik=accum(loglik,aux)

            if aux>logmax:
                logmax,ind=aux,hypot.count

            # Update the log-probability and accumulate them.
            hypot.logprob+=math.log1p(-ratefun(hypot.count))+logdens
            logsum=accum(logsum,hypot.logprob)

        if self.__alg__=='maxprod':

            loglik=logmax

            # Keep track of the most
            # likely hypotheses.
            self.__ind__.append(ind)

        stat=suffstat(*self.__param__)

        # Add a new hypothesis, which states that
        # the next segment is about to begin.
        self.__hypot__.append(struct(count=0,
                                     logprob=loglik,
                                     stat=stat,
                                     logconst=stat.logconst(),
                                     featfun=fun))

        logsum=accum(logsum,loglik)

        # Normalize the hypotheses so that
        # their probabilities sum to one.
        for hypot in self.__hypot__:
            hypot.logprob-=logsum

    def trim(self,minprob=1.0e-6,maxhypot=20):

        # Sort the hypotheses according to their probability.
        self.__hypot__.sort(key=lambda x:-x.logprob)

        # Store the indices of likely hypotheses.
        ind=[i for i,hypot in enumerate(self.__hypot__)
             if math.exp(hypot.logprob)>minprob]

        ind=ind[:maxhypot]

        if not ind:
            ind=[0]

        # Trim the hypotheses.
        self.__hypot__=[self.__hypot__[i] for i in ind]

        logsum=-numpy.inf

        # Normalize the hypotheses so that
        # their probabilities sum to one.
        for hypot in self.__hypot__:
            logsum=accum(logsum,hypot.logprob)
        for hypot in self.__hypot__:
            hypot.logprob-=logsum

    def segment(self):

        k=len(self.__ind__)

        # Find the most likely hypothesis.
        hypot=max(self.__hypot__,key=lambda x:x.logprob)

        count=hypot.count

        # Initialize the
        # segment boundaries.
        segbound=[k]

        # Find the best sequence
        # segmentation given all
        # the data so far.
        ind=k-1
        while ind>0:
            ind-=count
            segbound.append(ind)
            count=self.__ind__[ind-1]

        segbound.reverse()

        return segbound

    def state(self):

        # Iterate over the segmentation hypotheses.
        for hypot in self.__hypot__:
            yield hypot.count,math.exp(hypot.logprob)
