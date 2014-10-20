
import math,numpy

from numpy import linalg,random
from scipy import special

class suffstat(object):

    def __init__(self,mu,omega,sigma,eta):

        m,n=numpy.shape(mu)

        self.__size__=m,n

        # Allocate space for storing
        # the matrix of product statistics.
        self.__prod__=numpy.zeros([m+n,m+n])

        x=numpy.dot(omega,mu)

        # Initialize the statistics with the
        # parameters of the prior distribution.
        self.__prod__[:m,:m]=omega
        self.__prod__[:m,m:]=x
        self.__prod__[m:,:m]=x.transpose()
        self.__prod__[m:,m:]=numpy.dot(numpy.transpose(mu),x)+eta*sigma
        self.__weight__=eta

    def update(self,pred,resp):

        m,n=self.__size__

        if numpy.ndim(pred)>1:

            k,m=numpy.shape(pred)

            x=numpy.dot(numpy.transpose(pred),resp)

            # Update the statistics given a block of data.
            self.__prod__[:m,:m]+=numpy.dot(numpy.transpose(pred),pred)
            self.__prod__[:m,m:]+=x
            self.__prod__[m:,:m]+=x.transpose()
            self.__prod__[m:,m:]+=numpy.dot(numpy.transpose(resp),resp)
            self.__weight__+=k

        else:

            m=numpy.size(pred)

            x=numpy.outer(pred,resp)

            # Update the statistics given a single datum.
            self.__prod__[:m,:m]+=numpy.outer(pred,pred)
            self.__prod__[:m,m:]+=x
            self.__prod__[m:,:m]+=x.transpose()
            self.__prod__[m:,m:]+=numpy.outer(resp,resp)
            self.__weight__+=1

    def logconst(self):

        m,n=self.__size__

        d=numpy.diag(linalg.cholesky(self.__prod__))
        w=self.__weight__

        # Evaluate the log-normalization constant.
        return special.gammaln(0.5*(w-numpy.arange(n))).sum()\
               -n*(0.5*w)*math.log(0.5*w)-n*numpy.log(d[:m]).sum()\
               -w*numpy.log(d[m:]/math.sqrt(w)).sum()

    def param(self):

        m,n=self.__size__

        s=linalg.cholesky(self.__prod__).transpose()
        w=self.__weight__

        # Compute the parameters of the posterior distribution.
        return linalg.solve(s[:m,:m],s[:m,m:]),\
               numpy.dot(s[:m,:m].transpose(),s[:m,:m]),\
               numpy.dot(s[m:,m:].transpose(),s[m:,m:])/w,w

    def rand(self):

        m,n=self.__size__

        s=linalg.cholesky(self.__prod__).transpose()
        w=self.__weight__

        # Compute the parameters of
        # the posterior distribution.
        mu=linalg.solve(s[:m,:m],s[:m,m:])
        omega=numpy.dot(s[:m,:m].transpose(),s[:m,:m])
        sigma=numpy.dot(s[m:,m:].transpose(),s[m:,m:])/w
        eta=w

        # Simulate the marginal Wishart distribution.
        f=linalg.solve(numpy.diag(numpy.sqrt(2.0*random.gamma(
            (eta-numpy.arange(n))/2.0)))+numpy.tril(random.randn(n,n),-1),
            math.sqrt(eta)*linalg.cholesky(sigma).transpose())
        b=numpy.dot(f.transpose(),f)

        # Simulate the conditional Gauss distribution.
        a=mu+linalg.solve(linalg.cholesky(omega).transpose(),numpy.dot(
            random.randn(m,n),linalg.cholesky(b).transpose()))

        return a,b
