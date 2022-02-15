#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import erf, erfinv
from matplotlib import pyplot as plt
from forecasters.recalibration import (
  EWARecalibratedRegressionForecaster, MeanRecalibratedRegressionForecaster
)
from forecasters.calibration import quantile_calib_loss, pit_calib_loss

# We sample y from a Gaussian(mu, sigma)

# parameters
mu, sigma = 0.0, 1.0 # distribution over y
T = 500 # number of time steps of online learning
N = 20 # recalibrator discretizes probs into N intervals; best perf is 1/N
cal_eval_levels = [0.2, 0.4, 0.5, 0.6, 0.8] # measure calibration at these


# set up experiment
mu_raw, sigma_raw = 0.0, 2.0 # to generate uncalibrated forecasts

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

# helpers for CRPS loss
y_vals = np.arange(-2.0, 2.0, 0.05) # for CRPS integral approximation
p_raw_vals = [0.5*(1 + erf((y-mu_raw)/(sigma_raw*np.sqrt(2)))) for y in y_vals]

# run experiment
F_losses = np.zeros(T,)
Y = np.zeros(T,)
P = np.zeros(T,)
P_exp = np.zeros(T,)
P_raw = np.zeros(T,)
for t in range(T):
  if t % 100 == 0: print('Time: %05d/%05d' % (t, T))
  Y[t] = np.random.normal(mu, sigma) # sample Y from true distribution
  P_raw[t] = 0.5*(1 + erf((Y[t]-mu_raw)/(sigma_raw*np.sqrt(2)))) #uncalib fcst
  P[t] = R.predict(P_raw[t]) # theoretically justified recalibration
  P_exp[t] = R.expected_prediction(P_raw[t]) # no theory; better in practice

  # may be useful for debugging:
  # print('Y[t]', Y[t])
  # print('P_raw[t]', P_raw[t])
  # print('idx[t]', R._get_idx(P_raw[t]))
  # print('P_t[t]', P[t])
  # print('P_exp[t]', P_exp[t])
  # print('P_correct[t]',0.5*(1 + erf((Y[t]-mu)/(sigma*np.sqrt(2)))))
  # print()

  # compute crps loss
  # p_vals = np.array([R.predict(p_raw_val) for p_raw_val in p_raw_vals])
  p_vals = np.array([
    R.expected_prediction(p_raw_val) for p_raw_val in p_raw_vals
  ])
  F_losses[t] = crps_loss(p_vals, y_vals, Y[t])

  # pass observed outcome to recalibrator
  R.observe(Y[t], P_raw[t])

# plot the expected loss:
print('Plotting CRPS loss over time')
plt.subplot(211)
cum_losses = np.array([1/float(t+1) for t in range(T)]) * np.cumsum(F_losses)
plt.plot(range(T), cum_losses)
plt.xlabel('Time steps')
plt.ylabel('CRPS')

# plot calibration
print('Plotting calibration loss over time')
plt.subplot(212)
cum_cal_loss = np.array([
  pit_calib_loss(P_exp[:t], cal_eval_levels) for t in range(T)
])
plt.plot(range(5,T), cum_cal_loss[5:], color='black') # skip first 5 t
plt.xlabel('Time steps')
plt.ylabel('Calibration Error')

# this shows the p_raw -> p mapping of R(p_raw)
print('Recalibration Plot (raw, recalibrated, ideal):')
for i in range(1,10):
    p_raw = float(i)/10.
    y_raw = mu_raw + sigma_raw * np.sqrt(2) * erfinv(2*p_raw-1)
    p_true = 0.5*(1 + erf((y_raw-mu)/(sigma*np.sqrt(2))))
    p_pred = R.F_cal[R._get_idx(p_raw)+1].predict()
    print('%.4f, %.4f, %.4f' % (p_raw, p_pred, p_true))

print('Calibration Stats (Raw):')
cal_loss = 0
for p in cal_eval_levels:
    p_hat = np.sum([1 for p_t in P_raw if p_t <= p]) / T
    cal_loss += (p_hat - p)**2
    print(p, p_hat)
print('Loss: %f\n' % cal_loss)

print('Calibration Stats (Recalibrated):')
cal_loss = 0
for p in cal_eval_levels:
    p_hat = np.sum([1 for p_t in P if p_t <= p]) / T
    w_hat = np.sum([1 for p_t in P if p_t <= p])
    cal_loss += (p_hat - p)**2
    print(p, p_hat)
print('Loss: %f %f %f\n' % (
  cal_loss, quantile_calib_loss(P, cal_eval_levels), cum_cal_loss[-1]
)) 

print('Calibration Stats (Recalibrated-Exp):')
cal_loss = 0
for p in cal_eval_levels:
    p_hat = np.sum([1 for p_t in P_exp if p_t <= p]) / T
    cal_loss += (p_hat - p)**2
    print(p, p_hat) 
print('Loss: %f\n' % cal_loss) 

# print('CRPS Analysis:')
# # compute crps loss
# p_vals = np.array([np.mean(np.array([R.expected_prediction(p_raw_val) for _ in range(1000)]))
#   for p_raw_val in p_raw_vals])
# y_target = Y[t]
# n_vals = len(y_vals)
# crps_vals = (p_vals-np.array(y_vals >= y_target, dtype=np.int))**2
# delta_vals = np.array([y_vals[i+1] - y_vals[i] for i in range(n_vals-1)])
# print([(i, f.examples_seen) for i, f in enumerate(R.F_cal)])
# print(y_vals)
# print(p_raw_vals)
# print(p_vals)
# print(np.array(y_vals >= y_target, dtype=np.int))
# print(crps_vals)
# print(delta_vals)

# plt.show()
plt.savefig('calibration-gaussian.png')
