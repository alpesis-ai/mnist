########################################
MNIST (theano)
########################################

::

    main.py <- apps <- models
                    <- datasets
                    <- utils.py


Models:

::

    LogisticRegression -> MLP
                       -> CNN (LeNet)
                       -> Denoising AutoEncoder
                       -> Stacked Denoising AutoEncoder
                       -> Restricted Boltzmann Machines
                       -> Deep Belief Networks  <- Gibb Chain
                                                <- Hybrid Monte-Carlo Sampling
