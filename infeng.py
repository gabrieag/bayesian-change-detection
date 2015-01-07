import numpy as np
from numpy import linalg
from numpy import random
from scipy import special


class MatrixVariateNormalInvGamma(object):
    """Matrix-variate normal, inverse-gamma distribution.

    The matrix-variate normal, inverse-gamma distribution is the conjugate
    prior for a matrix-variate normal distribution. As a result the
    distribution can be used in Bayesian estimation of the location and scale
    parameters of the matrix-variate normal distribution.

    """

    def __init__(self, mu, omega, sigma, eta):

        # Get size of data.
        m, n = np.shape(mu)
        self.__m, self.__n = m, n

        # Check that the location parameter is a matrix of finite numbers.
        if not (np.ndim(mu) == 2 and
                np.shape(mu) == (m, n) and
                not np.isnan(mu).any() and
                np.isfinite(mu).all()):
            msg = 'The location parameter must be a matrix of finite numbers.'
            raise Exception(msg)

        # Check that the scale parameter is a symmetric, positive-definite
        # matrix.
        if not (np.ndim(omega) == 2 and
                np.shape(omega) == (m, m) and
                not np.isnan(omega).any() and
                np.isfinite(omega).all() and
                np.allclose(np.transpose(omega), omega) and
                linalg.det(omega) > 0.0):
            msg = 'The scale parameter must be a symmetric, positive-definite'
            msg += ' matrix.'
            raise Exception(msg)

        # Check that the dispersion parameter is a symmetric, positive-definite
        # matrix.
        if not (np.ndim(sigma) == 2 and
                np.shape(sigma) == (n, n) and
                not np.isnan(sigma).any() and
                np.isfinite(sigma).all() and
                np.allclose(np.transpose(sigma), sigma) and
                linalg.det(sigma) > 0.0):
            msg = 'The noise parameter must be a symmetric, positive-definite'
            msg += ' matrix.'
            raise Exception(msg)

        # Check that the shape parameter is a number greater than one minus the
        # number of degrees of freedom.
        if not (np.isscalar(eta) and
                not np.isnan(eta) and
                np.isfinite(eta) and eta > n - 1.0):
            msg = 'The shape parameter must be greater than one minus the'
            msg += ' degrees of freedom.'
            raise Exception(msg)

        # Allocate space for storing the matrix of product statistics.
        self.__prod = np.zeros([m + n, m + n])

        # Initialize the statistics with the parameters of the prior
        # distribution.
        x = np.dot(omega, mu)
        self.__prod[:m, :m] = omega
        self.__prod[:m, m:] = x
        self.__prod[m:, :m] = x.transpose()
        self.__prod[m:, m:] = np.dot(np.transpose(mu), x) + eta * sigma
        self.__weight = eta

    def update(self, X, Y):
        """Update the sufficient statistics given observed data.

        The sufficient statistics represent the only parameters required to
        describe the shape of the distribution. Due to conjugacy, the
        sufficient statistics also form the hyper-parameters of the
        distribution. In a Bayesian context, when the sufficient statistics are
        referred to as 'hyper-parameters', it implies the sufficient statistics
        reflect prior information about the distribution. After updating the
        distribution with observed data, the sufficient statistics refer to the
        posterior distribution.

        """

        # (Equation 5a, b)
        #
        #     | XX    XY |
        #     | YX    YY |
        #
        if np.ndim(X) > 1:
            k, m = np.shape(X)
            x = np.dot(np.transpose(X), Y)

            # Update the statistics given a block of data (in the following
            # order: XX, XY, YX, YY)
            self.__prod[:m, :m] += np.dot(np.transpose(X), X)
            self.__prod[:m, m:] += x
            self.__prod[m:, :m] += x.transpose()
            self.__prod[m:, m:] += np.dot(np.transpose(Y), Y)
            self.__weight += k

        else:
            m = np.size(X)
            x = np.outer(X, Y)

            # Update the statistics given a single datum.
            self.__prod[:m, :m] += np.outer(X, X)
            self.__prod[:m, m:] += x
            self.__prod[m:, :m] += x.transpose()
            self.__prod[m:, m:] += np.outer(Y, Y)
            self.__weight += 1

    def logconst(self):

        m, n = self.__m, self.__n

        # Note usage of the log-determinant 'trick':
        #
        #     log(det(A)) = 2*sum(log(diag(chol(A))))
        #
        d = np.diag(linalg.cholesky(self.__prod))
        w = self.__weight

        # Evaluate the log-normalization constant.
        # (Equation 8)
        return special.gammaln(0.5*(w - np.arange(n))).sum() - \
               n * np.log(d[:m]).sum() - \
               w * np.log(d[m:] / np.sqrt(w)).sum() - \
               n * (0.5 * w) * np.log(0.5 * w)

    def param(self):
        """Return parameters of the posterior distribution."""

        m = self.__m
        s = linalg.cholesky(self.__prod).transpose()
        w = self.__weight

        # Compute the parameters of the posterior distribution.
        return linalg.solve(s[:m, :m], s[:m, m:]), \
               np.dot(s[:m, :m].transpose(), s[:m, :m]), \
               np.dot(s[m:, m:].transpose(), s[m:, m:]) / w, \
               w

    def rand(self):

        m, n = self.__m, self.__n

        s = linalg.cholesky(self.__prod).transpose()
        w = self.__weight

        # Compute the parameters of the posterior distribution.
        mu = linalg.solve(s[:m, :m], s[:m, m:])
        omega = np.dot(s[:m, :m].transpose(), s[:m, :m])
        sigma = np.dot(s[m:, m:].transpose(), s[m:, m:]) / w
        eta = w

        # Simulate the marginal Wishart distribution.
        f = linalg.solve(np.diag(np.sqrt(2.0*random.gamma(
            (eta - np.arange(n))/2.0))) + np.tril(random.randn(n, n), -1),
                         np.sqrt(eta)*linalg.cholesky(sigma).transpose())
        b = np.dot(f.transpose(), f)

        # Simulate the conditional Gauss distribution.
        a = mu + linalg.solve(linalg.cholesky(omega).transpose(),
                              np.dot(random.randn(m, n),
                                     linalg.cholesky(b).transpose()))

        return a, b


class Bcdm():
    """Bayesian change detection model."""

    def __init__(self, mu=None, omega=None, sigma=None, eta=None,
                 alg='sumprod', ratefun=0.1, featfun=None, minprob=1.0e-6,
                 maxhypot=20):

        # The inference algorithm must be either sum-product or sum-product.
        assert alg in ['sumprod', 'maxprod']
        self.__alg__ = alg.lower()

        # Store number of dimensions in the predictor (independent/input
        # variable) and response (dependent/output variable) variables.
        self.__m = None
        self.__n = None

        self.__mu = None
        self.__omega = None
        self.__sigma = None
        self.__eta = None

        # Set prior for the location parameter.
        if mu is not None:
            self.__mu = mu

        # Set prior for the scale parameter.
        if omega is not None:
            self.__omega = omega

        # Set prior for the dispersion/noise parameter.
        if sigma is not None:
            self.__sigma = sigma

        # Set prior for the shape parameter.
        if eta is not None:
            self.__eta = eta

        # Ensure algorithm initialises on first call to update.
        self.__initialised = False

        # If 'maxhypot' is set to none, no hypotheses will be trimmed.
        assert maxhypot > 0 or not None
        assert minprob > 0
        self.__maximum_hypotheses = maxhypot
        self.__minimum_probability = minprob

        self.__hypotheses = list()
        self.__index = list()
        self.__probabilities = list()

        self.__featfun = featfun if callable(featfun) else lambda x: x
        self.__ratefun = ratefun if callable(ratefun) else lambda x: ratefun

    def __initialise_algorithm(self, m, n):

        # Ensure input dimensions are consistent.
        if self.__m is None:
            self.__m = m
        elif self.__m != m:
            msg = 'Expected %i dimensions in the predictor variable.' % m
            raise Exception(msg)

        # Ensure output dimensions are consistent.
        if self.__n is None:
            self.__n = n
        elif self.__n != n:
            msg = 'Expected %i dimensions in the response variable.' % n
            raise Exception(msg)

        # Set uninformative prior for the location parameter.
        if self.__mu is None:
            self.__mu = np.zeros([m, n])

        # Set uninformative prior for the scale parameter.
        if self.__omega is None:
            self.__omega = np.eye(m)

        # Set uninformative prior for the dispersion/noise parameter.
        if self.__sigma is None:
            self.__sigma = np.eye(n)

        # Set uninformative prior for the shape parameter.
        if self.__eta is None:
            self.__eta = n

        stat = MatrixVariateNormalInvGamma(self.__mu,
                                           self.__omega,
                                           self.__sigma,
                                           self.__eta)

        # Create the initial hypothesis, which states that the first segment is
        # about to begin.
        self.__update_count = 0
        self.__hypotheses = [{'index': 0,
                              'count': 0,
                              'log_probability': 0.0,
                              'distribution': stat,
                              'log_constant': stat.logconst()}]

    def __accum(self, x, y):
        return max(x, y) + np.log1p(np.exp(-abs(x - y)))

    def block_update(self, X, Y):

        for i in range(X.shape[0]):
            self.update(X[i, :], Y[i, :])

    def update(self, X, Y):

        # Initialise algorithm on first call to update. This allows the
        # algorithm to configure itself to the size of the first input/output
        # data if no hyper-parameters have been specified.
        if not self.__initialised:
            m = X.shape[1] if np.ndim(X) > 1 else X.size
            n = Y.shape[1] if np.ndim(Y) > 1 else Y.size
            self.__initialise_algorithm(m, n)
            self.__initialised = True

        # Get size of data.
        k = X.shape[0] if np.ndim(X) > 1 else X.size
        m, n = self.__m, self.__n

        loglik = -np.inf
        logmax = -np.inf
        logsum = -np.inf
        ind = np.nan

        self.__update_count += 1
        for hypotheses in self.__hypotheses:

            # Update the sufficient statistics.
            hypotheses['distribution'].update(self.__featfun(X), Y)

            # Compute the log-normalization constant after the update
            # (posterior parameter distribution).
            # (Equation 8)
            n_o = hypotheses['log_constant']
            n_k = hypotheses['log_constant'] = hypotheses['distribution'].logconst()

            # Evaluate the log-density of the predictive distribution.
            # (Equation 16)
            log_density = n_k - n_o - k * (0.5 * m * n) * np.log(2.0 * np.pi)

            # Increment the counter.
            hypotheses['count'] += 1

            # Accumulate the log-likelihood of the data.
            # (Equation 17)
            hazard = self.__ratefun(hypotheses['count'])
            aux = np.log(hazard) + log_density + hypotheses['log_probability']
            loglik = self.__accum(loglik, aux)

            # Keep track of the highest, log-likelihood.
            if aux > logmax:
                logmax, ind = aux, hypotheses['count']

            # Update and accumulate the log-probabilities.
            hypotheses['log_probability'] += np.log1p(-hazard) + log_density
            logsum = self.__accum(logsum, hypotheses['log_probability'])

        stat = MatrixVariateNormalInvGamma(self.__mu,
                                           self.__omega,
                                           self.__sigma,
                                           self.__eta)

        # Add a new hypothesis, which states that the next segment is about to
        # begin.
        self.__hypotheses.append({'index': self.__update_count,
                                  'count': 0,
                                  'log_probability': loglik,
                                  'distribution': stat,
                                  'log_constant': stat.logconst()})

        logsum = self.__accum(logsum, loglik)
        self.logsum = logsum

        # Normalize the hypotheses so that their probabilities sum to one.
        for hypothesis in self.__hypotheses:
            hypothesis['log_probability'] -= logsum

        # Automatically trim hypothesis on each update if requested.
        if self.__maximum_hypotheses is not None:
            self.trim(minprob=self.__minimum_probability,
                      maxhypot=self.__maximum_hypotheses)

        # In the max-product algorithm, keep track of the most likely
        # hypotheses.
        if self.__alg__ == 'maxprod':
            loglik = logmax
            self.__index.append(ind)

        # In the sum-product algorithm, keep track of the probabilities.
        else:
            iteration = list()
            for hypothesis in self.__hypotheses:
                iteration.append((hypothesis['count'],
                                  hypothesis['log_probability']))

            self.__probabilities.append(iteration)

    def trim(self, minprob=1.0e-6, maxhypot=20):

        if len(self.__hypotheses) <= maxhypot:
            return

        # Sort the hypotheses in decreasing log probability order.
        self.__hypotheses.sort(key=lambda dct: -dct['log_probability'])

        # Store the indices of likely hypotheses.
        minprob = np.log(minprob)
        index = [i for i, hypot in enumerate(self.__hypotheses)
                 if hypot['log_probability'] > minprob]

        # Trim the hypotheses.
        index = index[:maxhypot] if len(index) >= maxhypot else index
        self.__hypotheses = [self.__hypotheses[i] for i in index]

        # NOTE: This final ordering can preserve the original order of the
        #       hypotheses. Interestingly, the algorithm specified in update
        #       does not require that the hypotheses be ordered! This sort can
        #       safely be ignored.
        # self.__hypotheses.sort(key=lambda dct: dct['index'])

        # Normalize the hypotheses so that their probabilities sum to one.
        logsum = -np.inf
        for hypot in self.__hypotheses:
            logsum = self.__accum(logsum, hypot['log_probability'])
        for hypot in self.__hypotheses:
            hypot['log_probability'] -= logsum

    def segment(self):

        # In the max-product algorithm, the most likely hypotheses are
        # tracked. Recover the most likely segment boundaries by performing a
        # back-trace.
        if self.__alg__ == 'maxprod':

            # Find the most likely hypothesis.
            max_hypothesis = max(self.__hypotheses,
                                 key=lambda dct: dct['log_probability'])

            # Find the best sequence segmentation given all the data so far.
            segment_boundaries = [len(self.__index) - 1, ]
            index = segment_boundaries[0] - 1
            count = max_hypothesis['count'] - 1
            while index > 0:
                index -= count
                segment_boundaries.insert(0, index)
                count = self.__index[index - 1]

            return segment_boundaries

        # In the sum-product algorithm, the segment probabilities are
        # tracked. Recover the segment probabilities by formatting the stored
        # history.
        else:
            k = len(self.__probabilities)
            segment_probabilities = np.zeros((k + 1, k + 1))

            for i in range(len(self.__probabilities)):
                for (j, probability) in self.__probabilities[i]:
                    segment_probabilities[j, i + 1] = np.exp(probability)

            return segment_probabilities
