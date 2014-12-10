import math
import numpy
from numpy import linalg
from numpy import random
from scipy import special


class suffstat(object):

    def __init__(self, mu, omega, sigma, eta):

        m, n = numpy.shape(mu)
        self.__size__ = m, n

        # Check that the location parameter is a matrix of finite numbers.
        assert (numpy.ndim(mu) == 2 and
                numpy.shape(mu) == (m, n) and
                not numpy.isnan(mu).any() and
                numpy.isfinite(mu).all())

        # Check that the scale parameter is a symmetric, positive-definite
        # matrix.
        assert (numpy.ndim(omega) == 2 and
                numpy.shape(omega) == (m, m) and
                not numpy.isnan(omega).any() and
                numpy.isfinite(omega).all() and
                numpy.allclose(numpy.transpose(omega), omega) and
                linalg.det(omega) > 0.0)

        # Check that the dispersion parameter is a symmetric, positive-definite
        # matrix.
        assert (numpy.ndim(sigma) == 2 and
                numpy.shape(sigma) == (n, n) and
                not numpy.isnan(sigma).any() and
                numpy.isfinite(sigma).all() and
                numpy.allclose(numpy.transpose(sigma), sigma) and
                linalg.det(sigma) > 0.0)

        # Check that the shape parameter is a number greater than one minus the
        # number of degrees of freedom.
        assert (numpy.isscalar(eta) and
                not numpy.isnan(eta) and
                numpy.isfinite(eta) and eta > n - 1.0)

        # Allocate space for storing the matrix of product statistics.
        self.__prod__ = numpy.zeros([m + n, m + n])

        x = numpy.dot(omega, mu)

        # Initialize the statistics with the parameters of the prior
        # distribution.
        self.__prod__[:m, :m] = omega
        self.__prod__[:m, m:] = x
        self.__prod__[m:, :m] = x.transpose()
        self.__prod__[m:, m:] = numpy.dot(numpy.transpose(mu), x) + eta*sigma
        self.__weight__ = eta

    def update(self, pred, resp):

        m, n = self.__size__

        if numpy.ndim(pred) > 1:

            k, m = numpy.shape(pred)

            x = numpy.dot(numpy.transpose(pred), resp)

            # Update the statistics given a block of data.
            self.__prod__[:m, :m] += numpy.dot(numpy.transpose(pred), pred)
            self.__prod__[:m, m:] += x
            self.__prod__[m:, :m] += x.transpose()
            self.__prod__[m:, m:] += numpy.dot(numpy.transpose(resp), resp)
            self.__weight__ += k

        else:
            m = numpy.size(pred)
            x = numpy.outer(pred, resp)

            # Update the statistics given a single datum.
            self.__prod__[:m, :m] += numpy.outer(pred, pred)
            self.__prod__[:m, m:] += x
            self.__prod__[m:, :m] += x.transpose()
            self.__prod__[m:, m:] += numpy.outer(resp, resp)
            self.__weight__ += 1

    def logconst(self):

        m, n = self.__size__

        d = numpy.diag(linalg.cholesky(self.__prod__))
        w = self.__weight__

        # Evaluate the log-normalization constant.
        return special.gammaln(0.5*(w - numpy.arange(n))).sum() \
               -n*(0.5*w)*math.log(0.5*w) - n*numpy.log(d[:m]).sum() \
               -w*numpy.log(d[m:] / math.sqrt(w)).sum()

    def param(self):

        m, n = self.__size__

        s = linalg.cholesky(self.__prod__).transpose()
        w = self.__weight__

        # Compute the parameters of the posterior distribution.
        return linalg.solve(s[:m, :m], s[:m, m:]), \
               numpy.dot(s[:m, :m].transpose(), s[:m, :m]), \
               numpy.dot(s[m:, m:].transpose(), s[m:, m:]) / w,w

    def rand(self):

        m, n = self.__size__

        s = linalg.cholesky(self.__prod__).transpose()
        w = self.__weight__

        # Compute the parameters of the posterior distribution.
        mu = linalg.solve(s[:m, :m], s[:m, m:])
        omega = numpy.dot(s[:m, :m].transpose(), s[:m, :m])
        sigma = numpy.dot(s[m:, m:].transpose(), s[m:, m:]) / w
        eta = w

        # Simulate the marginal Wishart distribution.
        f = linalg.solve(numpy.diag(numpy.sqrt(2.0*random.gamma(
            (eta - numpy.arange(n))/2.0))) + numpy.tril(random.randn(n, n), -1),
                         math.sqrt(eta)*linalg.cholesky(sigma).transpose())
        b = numpy.dot(f.transpose(), f)

        # Simulate the conditional Gauss distribution.
        a = mu + linalg.solve(linalg.cholesky(omega).transpose(),
                              numpy.dot(random.randn(m, n),
                                        linalg.cholesky(b).transpose()))

        return a, b


class struct():
    def __init__(self, **arg):
        for key, val in arg.items():
            self.__dict__[key] = val


class Bcdm():
    """Bayesian change detection model."""

    def __init__(self, m, n, mu=None, omega=None, sigma=None, eta=None,
                 alg='sumprod', featfun=None):

        # The number of predictors and responses must be both positive integer
        # scalars.
        assert m > 0 and n > 0

        # The inference algorithm must be either sum-product or max-product.
        assert alg in ['sumprod', 'maxprod']

        self.__size__ = m, n
        self.__alg__ = alg.lower()

        # Set default values for the parameters.
        if mu is None:
            mu = numpy.zeros([m, n])
        if omega is None:
            omega = numpy.eye(m)
        if sigma is None:
            sigma = numpy.eye(n)
        if eta is None:
            eta = n

        self.__param__ = mu, omega, sigma, eta
        self.__hypot__ = []
        self.__ind__ = None

        stat = suffstat(*self.__param__)

        fun = featfun if callable(featfun) else lambda x: x

        # Create the initial hypothesis, which states that the first segment is
        # about to begin.
        self.__hypot__ = [struct(count=0,
                                 logprob=0.0,
                                 stat=stat,
                                 logconst=stat.logconst(),
                                 featfun=fun)]

        if self.__alg__ == 'maxprod':

            # The max-product algorithm involves keeping track of the most
            # likely hypotheses.
            self.__ind__ = []

    def sim(self, *pred):

        # NOTE: The previous definition is invalid in python 2.7:
        #           def sim(self, *pred, featfun=None):
        featfun = None

        m, n = self.__size__

        fun = featfun if callable(featfun) else lambda x: x

        # Generate the gain and noise parameters.
        gain, noise = suffstat(*self.__param__).rand()

        resp = []

        fact = linalg.cholesky(noise).transpose()

        # Given a set of predictor data, generate a corresponding set of
        # response data.
        for pred in pred:
            if numpy.ndim(pred) > 1:
                k, m = numpy.shape(pred)
                shape = [k, n]
            else:
                shape = [n]
            resp.append(fun(numpy.dot(pred, gain))
                        + numpy.dot(random.randn(*shape), fact))

        return resp

    def __accum(self, x, y):
        return max(x, y) + math.log1p(math.exp(-abs(x - y)))

    def update(self, pred, resp, featfun=None, ratefun=0.1):

        m, n = self.__size__

        # Deduce the number of points.
        if numpy.ndim(pred) > 1:
            k, _ = numpy.shape(pred)
        else:
            k = numpy.size(pred)

        fun = featfun if callable(featfun) else lambda x: x

        if not callable(ratefun):
            rate = float(ratefun)
            ratefun = lambda x: rate

        loglik = -numpy.inf
        logmax = -numpy.inf
        logsum = -numpy.inf

        ind = numpy.nan

        for i, hypot in enumerate(self.__hypot__):

            # Update the sufficient statistics.
            hypot.stat.update(hypot.featfun(pred), resp)

            # Compute the log-normalization constant of the posterior parameter
            # distribution.
            logconst = hypot.logconst
            hypot.logconst = hypot.stat.logconst()

            # Evaluate the log-density of the predictive distribution.
            logdens = hypot.logconst - logconst - \
                      k*(0.5*m*n)*math.log(2.0*math.pi)

            # Increment the counter.
            hypot.count += 1

            aux = math.log(ratefun(hypot.count)) + logdens + hypot.logprob

            # Accumulate the log-likelihood of the data.
            loglik = self.__accum(loglik, aux)

            if aux > logmax:
                logmax, ind = aux, hypot.count

            # Update the log-probability and accumulate them.
            hypot.logprob += math.log1p(-ratefun(hypot.count)) + logdens
            logsum = self.__accum(logsum, hypot.logprob)

        if self.__alg__ == 'maxprod':

            loglik = logmax

            # Keep track of the most likely hypotheses.
            self.__ind__.append(ind)

        stat = suffstat(*self.__param__)

        # Add a new hypothesis, which states that the next segment is about to
        # begin.
        self.__hypot__.append(struct(count=0,
                                     logprob=loglik,
                                     stat=stat,
                                     logconst=stat.logconst(),
                                     featfun=fun))

        logsum = self.__accum(logsum, loglik)

        # Normalize the hypotheses so that their probabilities sum to one.
        for hypot in self.__hypot__:
            hypot.logprob -= logsum

    def trim(self, minprob=1.0e-6, maxhypot=20):

        # Sort the hypotheses according to their probability.
        self.__hypot__.sort(key=lambda x: -x.logprob)

        # Store the indices of likely hypotheses.
        ind = [i for i, hypot in enumerate(self.__hypot__)
               if math.exp(hypot.logprob) > minprob]

        ind = ind[:maxhypot]

        if not ind:
            ind = [0]

        # Trim the hypotheses.
        self.__hypot__ = [self.__hypot__[i] for i in ind]

        logsum = -numpy.inf

        # Normalize the hypotheses so that their probabilities sum to one.
        for hypot in self.__hypot__:
            logsum = self.__accum(logsum, hypot.logprob)
        for hypot in self.__hypot__:
            hypot.logprob -= logsum

    def segment(self):

        k = len(self.__ind__)

        # Find the most likely hypothesis.
        hypot = max(self.__hypot__, key=lambda x: x.logprob)

        count = hypot.count

        # Initialize the segment boundaries.
        segbound = [k]

        # Find the best sequence segmentation given all the data so far.
        ind = k - 1
        while ind > 0:
            ind -= count
            segbound.append(ind)
            count = self.__ind__[ind - 1]

        segbound.reverse()

        return segbound

    def state(self):

        # Iterate over the segmentation hypotheses.
        for hypot in self.__hypot__:
            yield hypot.count, math.exp(hypot.logprob)


def filterdata(pred, resp, mu=None, omega=None, sigma=None, eta=None, **arg):

    k, m = numpy.shape(pred)
    k, n = numpy.shape(resp)

    # Create an inference engine of the appropriate size to run the sum-product
    # algorithm.
    bcdm = Bcdm(m, n, mu=mu, omega=omega, sigma=sigma, eta=eta, alg='sumprod')

    # Allocate space for storing the posterior probabilities of the
    # segmentation hypotheses.
    prob = numpy.zeros([k + 1, k + 1])

    # Initialize the probabilities.
    for j, alpha in bcdm.state():
        prob[j, 0] = alpha

    for i in range(k):

        # Update the segmentation hypotheses given the data, one point at a
        # time.
        bcdm.update(pred[i, :], resp[i, :], **arg)

        # Limit the number of hypotheses.
        bcdm.trim()

        # Update the probabilities.
        for j, alpha in bcdm.state():
            prob[j, i + 1] = alpha

    return prob


def segmentdata(pred, resp, mu=None, omega=None, sigma=None, eta=None, **arg):

    k, m = numpy.shape(pred)
    k, n = numpy.shape(resp)

    # Create an inference engine of the appropriate size to run the max-product
    # algorithm.
    bcdm = Bcdm(m, n, mu=mu, omega=omega, sigma=sigma, eta=eta, alg='maxprod')

    for i in range(k):

        # Update the segmentation hypotheses given the data, one point at a
        # time.
        bcdm.update(pred[i, :], resp[i, :], **arg)

        # Limit the number of hypotheses.
        bcdm.trim()

    # Backtrack to find the most likely segmentation of the sequence.
    return bcdm.segment()
