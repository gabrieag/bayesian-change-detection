Bayesian change detection
=========================

The Bayesian model-based change detection module implements a recursive algorithm for segmenting a sequence of real-valued input-output data. The segment boundaries are chosen under the assumption that, within each segment, the input-output data follow a multi-variate linear model. The parameters of the linear model (i.e. the coefficient matrix and the noise covariance matrix) are treated as random variables, thus resulting in a fully Bayesian model.

Sequence segmentation occurs online by recursively updating a set of segmentation hypotheses. Each hypothesis captures a particular belief about when the current segment started (i.e. how far back in time) given all the data so far. Every time new input-output data arrives, the hypotheses are updated to reflect this knowledge. The computational cost of each update step is kept constant via an approximation. The tradeoff between computational cost and approximation quality can be controlled with a tuning parameter.
