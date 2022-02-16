#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import erf, erfinv
from matplotlib import pyplot as plt
from sklearn.datasets import make_regression 
from sklearn.linear_model import BayesianRidge
from forecasters.recalibration import (
  EWARecalibratedRegressionForecaster, MeanRecalibratedRegressionForecaster
)
from forecasters.calibration import quantile_calib_loss, pit_calib_loss

# create a dataset
T = 100 # number of time steps of online learning
n_batch = 10 # take ten data points at a time
X, y = make_regression(
  n_samples=(T+1)*n_batch, n_features=100, n_informative=10, noise=0.0
)
y += np.random.poisson(size=y.shape)
N = 20 # recalibrator discretizes probs into N intervals; best perf is 1/N
cal_eval_levels = [0.2, 0.4, 0.5, 0.6, 0.8] # measure calibration at these

# define the model
ridge = BayesianRidge()

# train on initial batch
X_0, y_0 = X[:n_batch], y[:n_batch]
ridge.fit(X_0, y_0)

# construct the recalibrator
# eta = np.sqrt(2*np.log(N)/T) # this the theoretical learning rate
eta = 5 # this a practical effective learning rate
R = EWARecalibratedRegressionForecaster(N, eta)
# below is baseline that implements counting means in each of N bins
# R = MeanRecalibratedRegressionForecaster(N)

# we use the CRPS loss
# the CRPS loss works by discretizing the integral at y_vals
# and using Riemann sum approximation
def crps_loss(p_vals, y_vals, y_target):
  n_vals = len(y_vals)
  crps_vals = (p_vals-np.array(y_vals >= y_target, dtype=np.int))**2
  delta_vals = np.array([y_vals[i+1] - y_vals[i] for i in range(n_vals-1)])
  return np.sum(crps_vals[:n_vals-1] * delta_vals)

# run experiment
F_losses = np.zeros([T,n_batch])
P = np.zeros([T,n_batch])
P_exp = np.zeros([T,n_batch])
P_raw = np.zeros([T,n_batch])
for t in range(1,T):
  print('Time: %02d/%02d' % (t, T))

  # take batch
  X_t, y_t = X[n_batch*(t-1):n_batch*t], y[n_batch*(t-1):n_batch*t]

  # make predictions for batch
  mu_raw, sigma_raw = ridge.predict(X_t, return_std=True)

  # compute loss
  F_losses[t,:] = np.sqrt(np.sum((y_t - mu_raw)**2))

  # compute probabilities
  P_raw[t,:] = 0.5*(1 + erf((y_t-mu_raw)/(sigma_raw*np.sqrt(2)))) # uncalibr
  P[t,:] = [R.predict(pi) for pi in P_raw[t]]
  P_exp[t,:] = [R.expected_prediction(pi) for pi in P_raw[t]]

  # # compute crps loss # TODO:
  # # p_vals = np.array([R.predict(p_raw_val) for p_raw_val in p_raw_vals])
  # y_vals = np.arange(-2.0, 2.0, 0.05) # for CRPS integral approximation
  # p_raw_vals = [0.5*(1 + erf((y-mu_raw)/(sigma_raw*np.sqrt(2)))) for y in y_vals]
  # p_vals = np.array([
  #   R.expected_prediction(p_raw_val) for p_raw_val in p_raw_vals
  # ])
  # F_losses[t] = crps_loss(p_vals, y_vals, Y[t])

  # pass observed outcomes to recalibrator
  for yi, pi in zip(y_t, P_raw[t,:]):
    R.observe(yi, pi)

  # re-train using additional batch
  X_tt, y_tt = X[:t*n_batch], y[:t*n_batch]
  ridge.fit(X_tt, y_tt)

# average everything

# plot the expected loss:
print('Plotting CRPS loss over time')
plt.subplot(211)
cum_losses = np.array([1/float(t+1) for t in range(T)]) * np.cumsum(F_losses.mean(axis=1))
plt.plot(range(T), cum_losses)
plt.xlabel('Time steps')
plt.ylabel('L2 Loss')

# plot calibration
print('Plotting calibration loss over time')
plt.subplot(212)
cum_raw_loss = np.array([
  pit_calib_loss(P_raw[:t].flatten(), cal_eval_levels) for t in range(T)
])
cum_cal_loss = np.array([
  pit_calib_loss(P_exp[:t].flatten(), cal_eval_levels) for t in range(T)
])
plt.plot(range(1,T), cum_raw_loss[1:], color='red') # skip first 1 t
plt.plot(range(1,T), cum_cal_loss[1:], color='blue') # skip first 1 t
plt.xlabel('Time steps')
plt.ylabel('Calibration Error')
plt.legend(['raw', 'recal'])

# save the figures
# plt.show()
plt.savefig('recalibration-gaussian.png')
