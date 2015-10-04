#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from forecasters.external import EWAForecaster

# this reproduces example 4.1 from Cesa-Bianchi and Lugosi

# parameters
T = 900
N = 3

# construct experts
def expert(i, t):
  if t <= T/3:
    if i == 0:
      return 0
    elif i == 1:
      return 1
    elif i == 2:
      return 5
  elif T/3 < t <= 2*T/3:
    if i == 0:
      return 1
    elif i == 1:
      return 0
    elif i == 2:
      return 5
  elif 2*T/3 < t :
    if i == 0:
      return 1
    elif i == 1:
      return 0
    elif i == 2:
      return -1

# construct forecaster
eta = np.sqrt(2*np.log(N)/T)
F = EWAForecaster(N, eta)

# we use the L2 loss
def loss(p, y):
  return np.sqrt((p-y)**2)

# run experiment
e_losses = np.zeros((T,N))
F_losses = np.zeros(T,)
for t in xrange(T):
  e_losses[t,:] = np.array([expert(i,t) for i in xrange(N)])
  F_losses[t] = F.predict(e_losses[t,:])
  F.observe(e_losses[t,:])

# plot the losses
for i in xrange(N):
  cum_loss = np.array([1/float(t+1) for t in xrange(T)]) * np.cumsum(e_losses[:,i])
  plt.plot(range(T), cum_loss)
cum_loss = np.array([1/float(t+1) for t in xrange(T)]) * np.cumsum(F_losses)
plt.plot(range(T), cum_loss, color='black')
plt.legend(['1', '2', '3', '4'])
plt.show()