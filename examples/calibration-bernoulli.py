#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from forecasters.calibration import EWACalibratedForecaster, \
                                    calib_loss

# We sample y from Bernoulli(p)

# parameters
p = 0.8
T = 1000
N = 20

# construct forecaster
eta = np.sqrt(2*np.log(N)/T)
F = EWACalibratedForecaster(N, eta)

# we use the L2 loss
def loss(p, y):
  return np.sqrt((p-y)**2)

# run experiment
F_losses = np.zeros(T,)
Y = np.zeros(T,)
P = np.zeros(T,)
P_exp = np.zeros(T,)
for t in xrange(T):
  P[t] = F.predict()
  Y[t] = np.random.binomial(1, p)
  P_exp[t] = F.expected_prediction()

  F_losses[t] = loss(P[t], Y[t])

  F.observe(Y[t])

# plot the expected loss:
plt.subplot(211)
losses = np.array([(Y[t] - P_exp[t])**2 for t in xrange(T)])
cum_losses = np.array([1/float(t+1) for t in xrange(T)]) * np.cumsum(losses)
plt.plot(range(T), cum_losses)

# plot calibration
plt.subplot(212)
cum_cal_loss = np.array([calib_loss(Y[:t], P[:t], N) for t in xrange(T)])
plt.plot(range(T), cum_cal_loss, color='black')


plt.show()