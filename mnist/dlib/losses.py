""" Loss Functions

The general formulation for learning problem:

    f^* = min||f||^2 + sum( loss(y,f(x)) )

@zero_one_loss
"""


def zero_one_loss(y, y_hat):
    """
    Zero-One Loss: Standard loss function in classification.
    """

    if y == y_hat:
        return 1
    else
        return 0


def non_symmetric_loss(ham_hat, spam):
    """
    Non-Symmetric Losses: e.g. for spam classification.

    e.g.  L(ham_hat, spam) <= L(ham_hat, ham)
    """

    pass


def squared_loss(y, y_hat):
    """
    Squared Loss: standard loss function in regression.

    Calculation:
        L(y, y_hat) = (y_hat - y)^2

    Square loss is well-suited for the purpose of regression problems.

    However, it suffers from one critical flaw: outilers in the data
    (isolated points that are far from the desired target function) are
    punished very heavily by the squaring of the error.
    

    As a result, data must be filterred for outfilers first, or else
    the fit from this loss function may not be desirable.

    ref: http://courses.cms.caltech.edu/cs253/slides/cs253-14-GPs.pdf
    """

    loss = (y_hat, y)**2
    return loss


def negative_log_likelihood_loss():
    """
    """

    pass


def hinge_loss(y, y_hat):
    """
    Hinge loss:

        Loss(y, y_hat) = max(0, 1-y*y_hat)

    Hinge loss works well for its purposes in SVM as a classifier, since
    the more your violate the margin, the higher the penalty is.

    Hinge loss is not well-suited for regression-based problems as a result
    of its one-sided error.

    ref: http://courses.cms.caltech.edu/cs253/slides/cs253-14-GPs.pdf
    """

    hinge = 1 - y * y_hat
    if hinge > 0:
        return hinge
    else:
        return 0


def absolute_loss(y, y_hat):
    """
    Absolute Loss:
        
        Loss(y, y_hat) = | y - y_hat |

    Absolute loss is applicable to regression problems just like square loss,
    and it avoids the problem of weighting outliers too strongly by scaling
    the loss only linearly instead of quadratically by the error amount.
 
    ref: http://courses.cms.caltech.edu/cs253/slides/cs253-14-GPs.pdf
    """

    loss = abs(y - y_hat)

    return loss


def epsilon_insensitive_loss():
    """
    Exp-Insensitive Loss:

        loss(y, f(x)) = | y - f(x) |

    This loss function is ideal when small amounts of error (for example, in
    noisy data) are acceptable.

    It is identical in behavior to the absolute loss function, except that
    any points within some selected range epsilon incur no error at all.
    This error-free margin makes the loss function an ideal candidate
    for support vector regression. 

    ref: http://courses.cms.caltech.edu/cs253/slides/cs253-14-GPs.pdf
    """

    pass
