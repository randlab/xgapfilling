#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%
import numpy as np
import scipy.stats as stats
from scipy.interpolate import interp1d

# ==============================================================
# Define some functions
# ---------------------
def cdf_from_kde(kde, x0, x1, n=1000):
    """
    Computes the cdf (F) and and the inverse cdf (F^{-1}) of X restricted on the interval [x0, x1],
    from a kernel density estimate of data for X.

    :param kde   :  kernel density estimate (returned e.g. by scipy.stats.gaussian_kde)
    :param x0, x1:  min and max bound of the interval on which the functions are computed
    :param n     :  number of points in [x0, x1], used for estimating the returned functions
                        (scipy.interpolate.interp1d will be used)

    :return cdf, cdf_inv:
                    two functions approximating the cdf (F) and its inverse (F^{-1}) on [x0, x1];
                        note that with y0 = cdf(x0), y1 = cdf(x1), the conditional cdf and the
                        conditional inverse cdf of X | x0 < X < x1 can be approximated by
                        (cdf(.)-y0) / (y1-y0) and cdf_inv(y0 + (y1-y0)*.)

    """
    x = np.linspace(x0, x1, n)
    y0 = kde.integrate_box_1d(-np.inf, x0)
    y1 = kde.integrate_box_1d(-np.inf, x1)

    y = y0 + np.array([kde.integrate_box_1d(x0, xx) for xx in x])
    g = interp1d(x, y, bounds_error=False, fill_value=(y0, y1), assume_sorted=True)
    ginv = interp1d(y, x, bounds_error=False, fill_value=(x0, x1), assume_sorted=True)
    return g, ginv

def gpd_queue(x, kde, cdf, cdf_inv, x0, x1, pmarg, right_queue=True, method='MLE', plot_hill=False):
    """
    Fits a generalized Pareto distribution (GPD) to model a distribution queue.
    :param x           :    data (describing a random variable X)
    :param kde         :    kernel density estimate of the data (returned e.g. by scipy.stats.gaussian_kde)
    :param cdf, cdf_inv:    approximation of the cdf (F) and its inverse (F^{-1}) on [x0, x1]
    :param x0, x1      :    min and max bound of the interval on which the functions cdf, cdf_inv are defined
    :param pmarg       :    mass of the distribution queue to model with a GPD
    :param right_queue :    (bool) True for modeling the queue at rigth, False for modeling the queue at left
    :param method      :    (str) method to fit the shape parameter xi of the GPD:
                                'MLE' for maximum likelihood estimate
                                'Hill' for Hill estimator
    :param plot_hill   :    (bool) True for plotting the evolution of the Hill estimates
                                (if method='Hill'), False for not plotting
    :return xmarg, gdp     :
        xmarg:  the threshold on data values (quantile) corresponding to pmarg, i.e.
                    cdf(xmarg) = 1-pmarg if right_queue is set to True
                    cdf(xmarg) = pmarg if right_queue is set to False
        gpd:    GPD modeling the queue distribution:
                    if right_queue is True:
                        gpd is the distribution modeling X - xmarg | X > xmarg
                    if right_queue is False:
                        gpd is the distribution modeling xmarg - X | X < xmarg
                    the loc parameter is 0.0, the scale parameter is set such that
                    gpd.pdf(xmarg) = kde(xmarg), ensuring continuity, and the shape parameter xi
                    is fitted using the given method
                    Note:
                        gpd.args gives the fitted shape parameter xi,
                        gpd.kwds gives a dictionary with the loc and scale parameters
        Note that xmarg = None, gpd = None is returned if pmarg = 0 or
            - if cdf(x1) < 1-pmarg (if right_queue is True)
            - if cdf(x0) > pmarg (if right_queue is False)
    """
    # Initialize
    xmarg, gpd = None, None
    ok = True
    if pmarg == 0:
        ok = False
    elif pmarg > 0.:
        if right_queue:
            if cdf(x1) < 1-pmarg:
                ok = False
            else:
                xmarg = cdf_inv(1-pmarg)
                x_sub = x[x>xmarg] - xmarg
        else: # left queue
            if cdf(x0) > pmarg:
                ok = False
            else:
                xmarg = cdf_inv(pmarg)
                x_sub = xmarg - x[x<xmarg]

    if ok:
        # Set scale parameter of GPD, so that continuity is ensured
        s = pmarg/kde(xmarg) # [0]

        # Fit shape parameter of GPD
        if method == 'MLE':
            xi, ltmp, stmp = stats.genpareto.fit(x_sub, floc=0, fscale=s, method="MLE")
        elif method == 'Hill':
            x_sub_sorted = 1 + np.sort(x_sub)
            log_x_sub_sorted = np.log(x_sub_sorted)
            if plot_hill:
                # Tail index estimates using 2+k largest data value, k=0, 1, ...
                xi_vec = [np.mean(log_x_sub_sorted[-k+1:] - log_x_sub_sorted[-k]) for k in range(2,len(x_sub)+1)]
                plt.plot(xi_vec)
                xi = xi_vec[-1]
            else:
                xi = np.mean(log_x_sub_sorted[1:] - log_x_sub_sorted[0])

        # Set GPD
        gpd = stats.genpareto(xi, loc=0, scale=s)

    return xmarg, gpd

def pdf_model(kde, pmarg0, xmarg0, gpd0, pmarg1, xmarg1, gpd1):
    """
    Computes f_hat, the modeled pdf, based on the modelisation of the queue
    at left and/or right.

    :param kde: kernel density estimate
    :param pmarg0, xmarg0, gpd0:
                mass, quantile and GPD distribution modeling the left queue
                    (xmarg0 and gpd0 is None if the left queue is not modeled with a GPD)
    :param pmarg1, xmarg1, gpd1:
                mass, quantile and GPD distribution modeling the right queue
                    (xmarg1 and gpd1 is None if the right queue is not modeled with a GPD)

    :return f_hat:
        function modeling the pdf, defined as
           f_hat(x) = pmarg0*gpd0.pdf(xmarg0-x), if x < xmarg0
           f_hat(x) = kde(x),                    if xmarg0 <= x <= xmarg1
           f_hat(x) = pmarg1*gpd1.pdf(x-xmarg1), if xmarg1 < x
    """
    if gpd0 is None and gpd1 is None:
        # no queue modeled
        f_hat = kde

    elif gpd0 is None:
        # only the right queue is modeled with a GPD
        def f_hat(x):
            is_dim0 = np.asarray(x).ndim == 0
            x = np.atleast_1d(x)
            ind = np.where(x <= xmarg1)
            ind1 = np.where(x > xmarg1)
            y = np.zeros_like(x)
            y[ind] = kde(x[ind])
            y[ind1] = pmarg1*gpd1.pdf(x[ind1]-xmarg1)
            #y = np.array([kde(xx) if xx <= xmarg1 else pmarg1*gpd1.pdf(xx-xmarg1) for xx in x])
            if is_dim0:
                y = y[0]
            return y

    elif gpd1 is None:
        # only the left queue is modeled with a GPD
        def f_hat(x):
            is_dim0 = np.asarray(x).ndim == 0
            x = np.atleast_1d(x)
            ind = np.where(x >= xmarg0)
            ind0 = np.where(x < xmarg0)
            y = np.zeros_like(x)
            y[ind] = kde(x[ind])
            y[ind0] = pmarg0*gpd0.pdf(xmarg0-x[ind0])
            #y = np.array([kde(xx) if xx >= xmarg0 else pmarg0*gpd0.pdf(xmarg0-xx) for xx in x])
            if is_dim0:
                y = y[0]
            return y

    else:
        # both the left and right queues are modeled with a GPD
        def f_hat(x):
            is_dim0 = np.asarray(x).ndim == 0
            x = np.atleast_1d(x)
            ind = np.where(np.all((x >= xmarg0, x <= xmarg1), axis=0))
            ind0 = np.where(x < xmarg0)
            ind1 = np.where(x > xmarg1)
            y = np.zeros_like(x)
            y[ind] = kde(x[ind])
            y[ind0] = pmarg0*gpd0.pdf(xmarg0-x[ind0])
            y[ind1] = pmarg1*gpd1.pdf(x[ind1]-xmarg1)
            #y = np.array([pmarg0*gpd0.pdf(xmarg0-xx) if xx < xmarg0 else kde(xx) if xx <= xmarg1 else pmarg1*gpd1.pdf(xx-xmarg1) for xx in x])
            if is_dim0:
                y = y[0]
            return y

    return f_hat

def pdf_model_rvs(cdf_inv, pmarg0, xmarg0, gpd0, pmarg1, xmarg1, gpd1, size=1):
    """
    Draw samples from f_hat, where f_hat = pdf_model(kde, pmarg0, xmarg0, gpd0, pmarg1, xmarg1, gpd1),
    where kde is the kernel density estimate from which the cdf_inv is computed.
    at left and/or right.

    :param cdf_inv:
                inverse of the cdf used for drawing values in [xmarg0, xmarg1]
    :param pmarg0, xmarg0, gpd0:
                mass, quantile and GPD distribution modeling the left queue
                    (xmarg0 and gpd0 is None if the left queue is not modeled with a GPD)
    :param pmarg1, xmarg1, gpd1:
                mass, quantile and GPD distribution modeling the right queue
                    (xmarg1 and gpd1 is None if the right queue is not modeled with a GPD)

    :param size:
                samples size, number of value drawn

    :return x:  1d array of size 'size', samples drawn from f_hat
    """
    y = np.random.random(size=size)

    if gpd0 is None and gpd1 is None:
        # no queue modeled
        x = cdf_inv(y)

    elif gpd0 is None:
        # only the right queue is modeled with a GPD
        ind = np.where(y <= 1-pmarg1)
        ind1 = np.where(y > 1-pmarg1)
        x = np.zeros_like(y)
        x[ind] = cdf_inv(y[ind])
        x[ind1] = xmarg1 + gpd1.rvs(size=len(ind1[0]))

    elif gpd1 is None:
        # only the left queue is modeled with a GPD
        ind = np.where(y >= pmarg0)
        ind0 = np.where(y < pmarg0)
        x = np.zeros_like(y)
        x[ind] = cdf_inv(y[ind])
        x[ind0] = xmarg0 - gpd0.rvs(size=len(ind0[0]))

    else:
        # both the left and right queues are modeled with a GPD
        ind = np.where(np.all((y >= pmarg0, y <= 1-pmarg1), axis=0))
        ind0 = np.where(y < pmarg0)
        ind1 = np.where(y > 1-pmarg1)
        x = np.zeros_like(y)
        x[ind] = cdf_inv(y[ind])
        x[ind0] = xmarg0 - gpd0.rvs(size=len(ind0[0]))
        x[ind1] = xmarg1 + gpd1.rvs(size=len(ind1[0]))

    return x

def cdf_model(cdf, cdf_inv, pmarg0, xmarg0, gpd0, pmarg1, xmarg1, gpd1):
    """
    Computes F_hat, and F_hat^{-1}, the modeled cdf and its inverse, based on the
    modelisation of the queue at left and/or right.
    :param cdf, cdf_inv:
                approximation of the cdf (F) and its inverse (F^{-1}),
                    defined for input not in the queue
    :param pmarg0, xmarg0, gpd0:
                mass, quantile and GPD distribution modeling the left queue
                    (xmarg0 and gpd0 is None if the left queue is not modeled with a GPD)
    :param pmarg1, xmarg1, gpd1:
                mass, quantile and GPD distribution modeling the right queue
                    (xmarg1 and gpd1 is None if the right queue is not modeled with a GPD)

    :return F_hat, F_hat_inv:
        function modeling the cdf and its inverse, defined as
           F_hat(x) = pmarg0*(1 - gpd0.cdf(xmarg0-x)),     if x < xmarg0
           F_hat(x) = cdf(x),                       if xmarg0 <= x <= xmarg1
           F_hat(x) = (1-pmarg1) + pmarg1*gpd1.cdf(x-xmarg1), if xmarg1 < x
        and
           F_hat_inv(y) = xmarg0 - gpd0.ppf(1-y/pmarg0),            if y < pmarg0
           F_hat_inv(y) = cdf_inv(y),                               if pmarg0 <= y <= 1-pmarg1
           F_hat_inv(y) = xmarg1 + gpd1.ppf((y-(1-pmarg1))/pmarg1), if 1-pmarg1 < y
    """
    if gpd0 is None and gpd1 is None:
        # no queue modeled
        F_hat = cdf
        F_hat_inv = cdf_inv

    elif gpd0 is None:
        # only the right queue is modeled with a GPD
        def F_hat(x):
            is_dim0 = np.asarray(x).ndim == 0
            x = np.atleast_1d(x)
            ind = np.where(x <= xmarg1)
            ind1 = np.where(x > xmarg1)
            y = np.zeros_like(x)
            y[ind] = cdf(x[ind])
            y[ind1] = (1.0-pmarg1) + pmarg1*gpd1.cdf(x[ind1]-xmarg1)
            #y = np.array([cdf(xx) if xx <= xmarg1 else (1.0-pmarg1) + pmarg1*gpd1.cdf(xx-xmarg1) for xx in x])
            if is_dim0:
                y = y[0]
            return y

        def F_hat_inv(y):
            is_dim0 = np.asarray(y).ndim == 0
            y = np.atleast_1d(y)
            ind = np.where(y <= 1-pmarg1)
            ind1 = np.where(y > 1-pmarg1)
            x = np.zeros_like(y)
            x[ind] = cdf_inv(y[ind])
            x[ind1] = xmarg1 + gpd1.ppf((y[ind1]-(1-pmarg1))/pmarg1)
            #x = np.array([cdf_inv(yy) if yy <= pmarg1 else xmarg1 + gpd1.ppf((yy-(1-pmarg1))/pmarg1) for yy in y])
            if is_dim0:
                x = x[0]
            return x

    elif gpd1 is None:
        # only the left queue is modeled with a GPD
        def F_hat(x):
            is_dim0 = np.asarray(x).ndim == 0
            x = np.atleast_1d(x)
            ind = np.where(x >= xmarg0)
            ind0 = np.where(x < xmarg0)
            y = np.zeros_like(x)
            y[ind] = cdf(x[ind])
            y[ind0] = pmarg0*(1.0 - gpd0.cdf(xmarg0-x[ind0]))
            #y = np.array([cdf(xx) if xx >= xmarg0 else pmarg0*(1.0 - gpd0.cdf(xmarg0-xx)) for xx in x])
            if is_dim0:
                y = y[0]
            return y

        def F_hat_inv(y):
            is_dim0 = np.asarray(y).ndim == 0
            y = np.atleast_1d(y)
            ind = np.where(y >= pmarg0)
            ind0 = np.where(y < pmarg0)
            x = np.zeros_like(y)
            x[ind] = cdf_inv(y[ind])
            x[ind0] = xmarg0 - gpd0.ppf(1.0-y[ind0]/pmarg0)
            #x = np.array([cdf_inv(yy) if yy >= pmarg0 else xmarg0 - gpd0.ppf(1.0-yy/pmarg0) for yy in y])
            if is_dim0:
                x = x[0]
            return x

    else:
        # both the left and right queues are modeled with a GPD
        def F_hat(x):
            is_dim0 = np.asarray(x).ndim == 0
            x = np.atleast_1d(x)
            ind = np.where(np.all((x >= xmarg0, x <= xmarg1), axis=0))
            ind0 = np.where(x < xmarg0)
            ind1 = np.where(x > xmarg1)
            y = np.zeros_like(x)
            y[ind] = cdf(x[ind])
            y[ind0] = pmarg0*(1.0 - gpd0.cdf(xmarg0-x[ind0]))
            y[ind1] = (1.0-pmarg1) + pmarg1*gpd1.cdf(x[ind1]-xmarg1)
            # y = np.array([pmarg0*(1.0 - gpd0.cdf(xmarg0-xx)) if xx < xmarg0 else cdf(xx) if xx <= xmarg1 else (1.0-pmarg1) + pmarg1*gpd1.cdf(xx-xmarg1) for xx in x])
            if is_dim0:
                y = y[0]
            return y

        def F_hat_inv(y):
            is_dim0 = np.asarray(y).ndim == 0
            y = np.atleast_1d(y)
            ind = np.where(np.all((y >= pmarg0, y <= 1-pmarg1), axis=0))
            ind0 = np.where(y < pmarg0)
            ind1 = np.where(y > 1-pmarg1)
            x = np.zeros_like(y)
            x[ind] = cdf_inv(y[ind])
            x[ind0] = xmarg0 - gpd0.ppf(1.0-y[ind0]/pmarg0)
            x[ind1] = xmarg1 + gpd1.ppf((y[ind1]-(1-pmarg1))/pmarg1)
            #x = np.array([xmarg0 - gpd0.ppf(1.0-yy/pmarg0) if yy < pmarg0 else cdf_inv(yy) if yy <= pmarg1 else xmarg1 + gpd1.ppf((yy-(1-pmarg1))/pmarg1) for yy in y])
            if is_dim0:
                x = x[0]
            return x

    return F_hat, F_hat_inv
# ==============================================================
