Introduction
============

In this lab assignment, we will implement a variational algorithm for Bayesian PCA. Unlike regular PCA based on
maximization of retained variance or minimization of projection error (see Bishop, 12.1.1 and 12.1.2), probabilistic
PCA defines a proper density model over observed and latent variables. We will work with a fully Bayesian model
this time, which is to say that we will put priors on our parameters and will be interested in learning the posterior over
those parameters. Bayesian methods are very elegant, but require a shift in mindset: we are no longer looking for a
point estimate of the parameters (as in maximum likelihood or MAP), but for a full posterior distribution.
The integrals involved in a Bayesian analysis are usually analytically intractable, so that we must resort to approxima-
tions. In this lab assignment, we will implement the variational method described in Bishop99. Chapters 10 and 12 of
the PRML book contain additional material that may be useful when doing this exercise.
•
[Bishop99] Variational Principal Components, C. Bishop, ICANN 1999 -
http://research.microsoft.com/pubs/67241/bishop-vpca-icann-99.pdf