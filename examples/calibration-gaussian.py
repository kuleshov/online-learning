#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import erf
from matplotlib import pyplot as plt
from forecasters.recalibration import EWARecalibratedForecaster
from forecasters.calibration import quantile_calib_loss

# We sample y from Bernoulli(p)

# parameters
mu, sigma = 0.0, 1.0
T = 4000
N = 20

# we use the CRPS loss
def crps_loss(p_vals, y_vals, y_target):
  n_vals = len(y_vals)
  crps_vals = (p_vals-np.array(y_vals >= y_target, dtype=np.int))**2
  delta_vals = np.array([y_vals[i+1] - y_vals[i] for i in range(n_vals-1)])
  return np.sum(crps_vals[:n_vals-1] * delta_vals)

# helpers for CRPS loss
mu_raw, sigma_raw = 0.0, 2.0 # uncalibrated forecast
y_vals = np.arange(-2.0, 2.0, 0.05)
p_raw_vals = [0.5*(1 + erf((y-mu_raw)/(sigma_raw*np.sqrt(2)))) for y in y_vals]

# construct forecaster
eta = np.sqrt(2*np.log(N)/T)
F = EWARecalibratedForecaster(N, eta)

# run experiment
F_losses = np.zeros(T,)
Y = np.zeros(T,)
P = np.zeros(T,)
P_exp = np.zeros(T,)
P_raw = np.zeros(T,)
for t in range(T):
  if t % 100 == 0: print(t)
  Y[t] = np.random.normal(mu, sigma)
  p_raw = 0.5*(1 + erf((Y[t]-mu_raw)/(sigma_raw*np.sqrt(2))))
  P[t] = F.predict(p_raw)
  P_exp[t] = F.expected_prediction(p_raw)
  P_raw[t] = p_raw

  # compute crps loss
  # p_vals = np.array([F.predict(p_raw_val) for p_raw_val in p_raw_vals])
  p_vals = np.array([F.expected_prediction(p_raw_val) for p_raw_val in p_raw_vals])
  F_losses[t] = crps_loss(p_vals, y_vals, Y[t])

  F.observe(Y[t], p_raw)

print('test0')
# plot the expected loss:
plt.subplot(211)
cum_losses = np.array([1/float(t+1) for t in range(T)]) * np.cumsum(F_losses)
plt.plot(range(T), cum_losses)
print('test1')

# plot calibration
plt.subplot(212)
cum_cal_loss = np.array([quantile_calib_loss(P[:t], 10) for t in range(T)])
plt.plot(range(5,T), cum_cal_loss[5:], color='black')

print('Calibration Stats (Raw):')
cal_loss = 0
for i in range(1,10):
    p = float(i)/10.
    p_hat = np.sum([1 for p_t in P_raw if p_t <= p]) / T
    cal_loss += (p_hat - p)**2
    print(p, p_hat)
print('Loss: %f\n' % cal_loss)

print('Calibration Stats (Recalibrated):')
cal_loss = 0
for i in range(1,10):
    p = float(i)/10.
    p_hat = np.sum([1 for p_t in P if p_t <= p]) / T
    w_hat = np.sum([1 for p_t in P if p_t <= p])
    cal_loss += w_hat * (p_hat - p)**2
    print(p, p_hat)
print('Loss: %f %f %f\n' % (cal_loss, quantile_calib_loss(P, 10), cum_cal_loss[-1])) 

print('Calibration Stats (Recalibrated-Exp):')
cal_loss = 0
for i in range(1,10):
    p = float(i)/10.
    p_hat = np.sum([1 for p_t in P_exp if p_t <= p]) / T
    cal_loss += (p_hat - p)**2
    print(p, p_hat)
print('Loss: %f\n' % cal_loss) 

print('CRPS Analysis:')
# compute crps loss
p_vals = np.array([np.mean(np.array([F.expected_prediction(p_raw_val) for _ in range(1000)]))
  for p_raw_val in p_raw_vals])
y_target = Y[t]
n_vals = len(y_vals)
crps_vals = (p_vals-np.array(y_vals >= y_target, dtype=np.int))**2
delta_vals = np.array([y_vals[i+1] - y_vals[i] for i in range(n_vals-1)])
print([(i, f.examples_seen) for i, f in enumerate(F.F_cal)])
print(y_vals)
print(p_raw_vals)
print(p_vals)
print(np.array(y_vals >= y_target, dtype=np.int))
print(crps_vals)
print(delta_vals)
# return np.sum(crps_vals[:n_vals-1] * delta_vals)

# plt.show()
plt.savefig('calibration-gaussian.png')
