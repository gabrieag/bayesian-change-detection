
import heapq,math,numpy

from numpy import linalg,random

# Import the module-specific class.
from __util__ import suffstat

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
            eta=self.__param[3]
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
        self.__hypot__=[(0.0,0,stat.logconst(),stat,fun)]

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

    def update(self,pred,resp,featfun=None,ratefun=0.1,maxhypot=10):

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

        mode,ind=-numpy.inf,numpy.nan
        loglik=-numpy.inf

        for i,(logprob,count,logconst,stat,featfun) in enumerate(self.__hypot__):

            # Update the sufficient statistics
            # for the current hypothesis.
            stat.update(featfun(pred),resp)

            logdens=logconst+k*(0.5*m*n)*math.log(2.0*math.pi)

            # Evaluate the log-density of
            # the predictive distribution.
            logconst=stat.logconst()
            logdens=logconst-logdens

            count+=1

            # Increment the log-likelihood of the data.
            aux=math.log(ratefun(count))+logdens+logprob
            loglik=max(loglik,aux)+math.log1p(math.exp(-abs(loglik-aux)))

            if aux>mode:
                mode,ind=aux,count

            # Update the log-probability of
            # the current hypothesis given the data.
            logprob+=math.log1p(-ratefun(count))+logdens

            self.__hypot__[i]=logprob,count,logconst,stat,featfun

        # The hypotheses have changed, and this may
        # have broken the invariance of the priority
        # queue. Make sure to restore it.
        heapq.heapify(self.__hypot__)

        if self.__alg__=='maxprod':

            # Keep track of the most
            # likely hypotheses.
            loglik=mode
            self.__ind__.append(ind)

        stat=suffstat(*self.__param__)

        # Create a new hypothesis, which states
        # that the next segment is about to begin.
        hypot=loglik,0,stat.logconst(),stat,fun

        # Insert this new hypothesis into the priority queue.
        heapq.heappush(self.__hypot__,hypot)

        # Limit the number of hypotheses by
        # filtering out the least likely ones.
        for i in range(len(self.__hypot__)-maxhypot):
            hypot=heapq.heappop(self.__hypot__)

        # Retrieve the most likely hypothesis.
        hypot,=heapq.nlargest(1,self.__hypot__)
        logprob,count,logconst,stat,featfun=hypot

        # Normalize the hypothesis log-probabilities.
        aux=logprob+math.log(sum(math.exp(p-logprob) for p,k,c,s,f in self.__hypot__))
        for i,(logprob,count,logconst,stat,featfun) in enumerate(self.__hypot__):
            self.__hypot__[i]=logprob-aux,count,logconst,stat,featfun

    def segment(self):

        k=len(self.__ind__)

        # Retrieve the most likely hypothesis.
        hypot,=heapq.nlargest(1,self.__hypot__)
        logprob,count,logconst,stat,featfun=hypot

        # Initialize the
        # segment boundaries.
        segbound=[k]

        # Find the best segmentation
        # given all the data so far.
        ind=k-1
        while ind>0:
            ind-=count
            segbound.append(ind)
            count=self.__ind__[ind-1]

        segbound.reverse()

        return segbound

    def state(self,param=False):

        if param:

            # Iterate over the segmentation hypotheses, and return
            # the segment-specific feature functions and parameters.
            for logprob,count,logconst,stat,featfun in self.__hypot__:
                yield count,math.exp(logprob),featfun,stat.param()

        else:

            # Iterate over the segmentation hypotheses.
            for logprob,count,logconst,stat,featfun in self.__hypot__:
                yield count,math.exp(logprob)
