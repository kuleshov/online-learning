#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from forecasters.external import EWAForecaster

# We sample y from Bernoulli(p)
# Expert 1 predicts p - d1 + e
# Expert 2 predicts p - d2 + e
# Expert 3 predicts p + e,
# where e is sampled from a Normal(0,sigma)

# parameters
p = 0.8
d = [0.2, 0.3, 0.0]
sigma = 0.01
T = 12000

# construct experts
def expert(i):
  e = np.random.normal(0, sigma)
  return p - d[i] + e

# construct forecaster
N = len(d)
eta = np.sqrt(2*np.log(N)/T)
F = EWAForecaster(N, eta)

# we use the L2 loss
def loss(p, y):
  return np.sqrt((p-y)**2)

# run experiment
e_losses = np.zeros((T,N))
F_losses = np.zeros(T,)
Y = np.zeros(T,)
e_pred = np.zeros((T,N))
for t in xrange(T):
  e_t = np.array([expert(i) for i in xrange(N)])
  e_pred[t,:] = e_t
  p_t = F.predict(e_t)

  y_t = np.random.binomial(1, p)
  Y[t] = y_t

  e_losses[t,:] = loss(e_t, 3*[y_t])
  F_losses[t] = loss(p_t, y_t)

  F.observe(e_losses[t,:])

# plot the losses
for i in xrange(N):
  cum_loss = np.array([1/float(t+1) for t in xrange(T)]) * np.cumsum(e_losses[:,i])
  plt.plot(range(T), cum_loss)
cum_loss = np.array([1/float(t+1) for t in xrange(T)]) * np.cumsum(F_losses)
plt.plot(range(T), cum_loss, color='black')
plt.legend(['1', '2', '3', 'F'])
plt.show()