# -*- coding: utf-8 -*-
"""
Created 2017

This code explores the rise-time of the type Ia SNe
in the PTF sample. It calculates explosion times and
the shape of the light-curve in terms of the Zheng+17 paper
analytical equation.

@author: semelipap
"""

import numpy as np
import matplotlib

matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
from scipy import stats
import glob
import iminuit
from astropy.io import ascii
from probfit import Chi2Regression
import os

import probfit


# Patch the plotting function to use more intermediate steps:
def draw_pdf_with_midpoints(f, arg, x, ax=None, scale=1.0, normed_pdf=False,
                            **kwds):
    ax = plt.gca() if ax is None else ax
    arg = probfit.plotting.parse_arg(f, arg, 1) if isinstance(arg,
                                                              dict) else arg

    x = np.linspace(x.min(), x.max(), num=500)
    yf = probfit.plotting.vector_apply(f, x, *arg)

    if normed_pdf:
        normed_factor = sum(yf)
        yf /= normed_factor
    yf *= scale

    ax.plot(x, yf, **kwds)
    return x, yf


def draw_x2(self, minuit=None, ax=None, parmloc=(0.05, 0.95), print_par=True,
            args=None, errors=None, grid=True, parts=False):
    '''
    Drawing function for plotting.
    '''
    data_ret = None
    error_ret = None
    total_ret = None
    part_ret = []

    ax = plt.gca() if ax is None else ax

    arg, error = probfit.plotting._get_args_and_errors(self, minuit, args,
                                                       errors)

    x = self.x
    y = self.y
    data_err = self.error

    data_ret = x, y
    if data_err is None:
        ax.plot(x, y, '+')
        err_ret = (np.ones(len(self.x)), np.ones(len(self.x)))
    else:
        ax.errorbar(x, y, data_err, fmt='.', color='k')
        err_ret = (data_err, data_err)
    draw_arg = [('lw', 2)]
    draw_arg.append(('color', 'r'))

    total_ret = draw_pdf_with_midpoints(self.f, arg, x, ax=ax, **dict(draw_arg))

    ax.grid(grid)

    txt = probfit.plotting._param_text(probfit.plotting.describe(self), arg,
                                       error)

    chi2 = self(*arg)
    txt += u'chi2/ndof = %5.4g(%5.4g/%d)' % (chi2 / self.ndof, chi2, self.ndof)

    if parts:
        f_parts = getattr(self.f, 'parts', None)
        if f_parts is not None:
            for p in f_parts():
                tmp = draw_pdf_with_midpoints(p, arg, x, ax=ax,
                                              **dict(draw_arg))
                part_ret.append(tmp)

    if print_par:
        ax.text(parmloc[0], parmloc[1], txt, ha='left', va='top',
                transform=ax.transAxes)

    return (data_ret, error_ret, total_ret, part_ret)


probfit.plotting.draw_pdf_with_midpoints = draw_pdf_with_midpoints
probfit.plotting.draw_x2 = draw_x2
'''
First import the aligned light-curves as given by the template_maker_classes.py
Want not only the points from the restrictive -17 but also further back in
time where that is possible.
 '''
os.system('mkdir -p forced_onrise')
#os.system('rm forced_onrise/*.png')  # Remove old plots


def polynomial(time, epsilon, a, C, t_e):
    '''
    A polynomial of degree 2 + epsilon
    '''
    if time < t_e:
        return C
    else:
        return C + a * (time - t_e) ** (2 + epsilon)


def zheng(time, A, t0, tb, s, alpha_d, alpha_r):
    '''
    The analytical function to describe a light-curve
    presented in Zheng and Filippenko 2017.

    The parameters are:

    A = A' scaling factor
    s  smoothing factor
    tb  breaktime
    t0 first light

    alpha_d, alpha_r shape parameters

    We keep the breaktime, tb and alpha_r fixed as
    suggested by the paper.
    '''

    t_ = (time - t0) / tb
    t = np.abs(t_)

    val = A * t ** -alpha_r * (1 + t ** (s * alpha_d)) ** (-2 / s)
    return np.select([t_ > 0.], [val])
    if t > 0:
        return
    else:
        return 0.


# First import all the files, fluxes and times of maximum
data_list = glob.glob('Processed_data/forced_fluxes/*')
more_info = ascii.read('lc_info_forced/info_f.txt') # Max date

keys = ['A', 'alpha_d', 't0', 's', 'alpha_r', 'tb']
# Open file to write down results
f = open('analytical_results.txt', 'w')
f.write('#name ')
for k in keys:
    f.write(' ')
    f.write(' '.join((k, k + '_err_low', k + '_err_hi')))
f.write('\n')

t_0 = []
s = []
for filename in data_list:
    name = filename.split('/')[-1].split('_')[0]
    lightcurve = ascii.read(filename, data_start=1, delimiter=' ')
    # data_end excludes the last line (indicating
    # number of rows)
    jd = lightcurve.field('jd')
    flux = np.asarray(lightcurve.field('Flux'))
    norm = flux.max()
    flux = flux / norm
    flux_err = lightcurve.field('flux_err1') / norm
    flux_err2 = lightcurve.field('flux_err2') / norm

    # Now we need to align the light-curves, by getting the jd_max from a file
    name_info = more_info['name']
    tmax = more_info['t_max'][np.where(name_info == name)]
    try:
        jd = np.asarray(jd) - tmax
    except ValueError:
        continue

    jd_valid = jd

    flux_valid = flux
    flux_err_valid = flux_err

    # Require at least 10 points with S/N over 5 for good fit
    # We want more data points than number of parameters for a valid fit

    if np.sum(flux_valid / flux_err_valid > 10) < 5:
        continue

    chi2 = Chi2Regression(zheng, jd_valid, flux_valid, error=flux_err_valid)
    m = iminuit.Minuit(chi2, A=1, error_A=0.1,
                       t0=-20, error_t0=1, limit_t0=(-30, -5),
                       tb=20.4, fix_tb=True, error_tb=1,
                       alpha_r=2.1, limit_alpha_r=(1, 3),
                       fix_alpha_r=True,
                       alpha_d=-2.52, error_alpha_d=0.15,
                       limit_alpha_d=(-4., -1.),
                       s=1.3, error_s=0.15, limit_s=(0.3, 3),
                       pedantic=True, print_level=0, errordef=1.0)
    m.set_strategy(2)
    m.migrad()
    chi2_ = chi2

    print name
    print 'Migrad ok: ', m.migrad_ok()
    if not m.migrad_ok():
        continue

    m.minos()  # It will exclude the fixed parameters
    "A, t0, tb, s, alpha_d, alpha_r"
    args = [np.random.random((1,3))]*6

    N_t, N_alpha = 200, 199
    t0_, alphad_ = np.meshgrid(np.linspace(-30, -5, N_t),
                               np.linspace(-4., -1., N_alpha))
    args = [np.full((N_alpha, N_t), m.values['A']),
            t0_,
            np.full((N_alpha, N_t), m.values['tb']),
            np.full((N_alpha, N_t), m.values['s']),
            alphad_,
            np.full((N_alpha, N_t), m.values['alpha_r'])]

    chi = (zheng(jd_valid[:, None, None], *args) - flux_valid[:, None, None]) / flux_err_valid[:, None, None]
    chi2 = np.sum(chi * chi, axis=0) - m.fval
    np.save('analytical_sum_contours/{}.npy'.format(name), chi2)


    # Collect all the Minos errors computed so far:
    errors = dict(m.get_merrors())

    f.write(name)

    for k in keys:
        f.write(' ')
        val = m.values[k]
        if k in errors:
            lower = errors[k]['lower']
            upper = errors[k]['upper']
        else:
            lower = upper = 0.
        f.write('{:.3f} {:.3f} {:.3f}'.format(val, lower, upper))

    f.write('\n')
    f.flush()
    #Plotting
    chi2_.draw(m)
    plt.title(name)
    plt.xlabel('Time wrt maximum')
    plt.ylabel('Normalised flux')
    plt.savefig('forced_analytical/{}.png'.format(name))
    plt.close()

    plt.figure()
    x = np.arange(-70, 200, step=0.3)
    y = [zheng(x_i, **m.values) for x_i in x]
    plt.plot(x, y, color='orange')
    plt.errorbar(jd_valid, flux_valid, yerr=flux_err_valid, fmt='+', color='k')
    plt.xlim(-70, 200)
    plt.ylim(-0.2, 1.2)
    plt.vlines(m.values['t0'], -0.1, 1, linestyles=':', color='r')
    plt.vlines(
            [m.values['t0'] - m.errors['t0'], m.values['t0'] + m.errors['t0']],
            -0.1, 1, linestyles=':', color='orange')
    plt.text(m.values['t0'], -0.18, '$t_0 = {:.1f}$'.format(m.values['t0']),
             horizontalalignment='center')
    plt.xlabel('Time wrt max (days)')
    plt.ylabel('Normalised flux')
    plt.title(name)
    plt.savefig('analytical_plots/{}.png'.format(name))
    plt.close()

    # plt.figure()
    # m.draw_mnprofile('t0')
    plt.figure('contours')
    try:
        m.draw_contour('t0', 'alpha_d')#, numpoints=100, nsigma=2, sigma_res=4)
        plt.xlabel('$t_0$')
        plt.ylabel(r'$\alpha_d$')
        #plt.savefig('analytical_contours/{}.png'.format(name))
    except:
        pass
        t_0.append(m.values['t0'])
        s.append(m.values['s'])

f.close()
print(len(t_0))
plt.hist(t_0, bins='auto', histtype='step')
plt.xlabel('Time of explosion, t_0')
plt.ylabel('Number of supernovae')

plt.figure()
plt.hist(s, bins='auto', histtype='step')
plt.xlabel('Transition parameter')
plt.ylabel('Number of supernovae')

plt.show()
