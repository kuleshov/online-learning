#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from forecaster import Forecaster
from internal import EWAInternalForecaster

class EWACalibratedForecaster(Forecaster):
  """Produces calibrated estimates by minimizng internal regret"""
  def __init__(self, N, eta=None):
    super(EWACalibratedForecaster, self).__init__(N+1)
    self.F_int = EWAInternalForecaster(N+1, eta)
    self.e = np.array([float(i)/N for i in range(N+1)])
    self.examples_seen = 0

  def predict(self):
    w = self.F_int.weights.reshape(self.N,)
    ind = np.random.multinomial(1, w)
    return ind.dot(self.e)

  def expected_prediction(self):
    w = self.F_int.weights.reshape(self.N,)
    return w.T.dot(self.e)

  def observe(self, y_t):
    l = np.array([(y_t - e)**2 for e in self.e])
    self.F_int.observe(l)
    self.examples_seen += 1

  @property
  def weights(self):
    return self.F_int.weights

class RunningAverageForecaster(Forecaster):
  """Baseline producing estimates by taking the average of inputs"""
  def __init__(self, N):
    super(RunningAverageForecaster, self).__init__(N+1)
    self.input_history = []
    self.examples_seen = 0

  def predict(self):
    if not self.input_history:
      return 0.0
    else:
      return np.mean(np.array(self.input_history))

  def expected_prediction(self):
    return self.predict()

  def observe(self, y_t):
    self.input_history.append(y_t)
    self.examples_seen += 1

# ----------------------------------------------------------------------------
# Helpers

def calib_loss(Y, P, N):
  loss = 0.0
  T = len(Y)
  if T == 0: return 0

  for i in range(N+1):
    p = float(i)/N
    w_i = np.sum([p_t for p_t in P if p_t == p])
    y_i = np.sum([p_t*y_t for p_t, y_t in zip(P,Y) if p_t == p])
    rho_i = y_i / w_i if w_i else 0.0
    loss += w_i*(rho_i-p)**2

  return loss / T

def quantile_calib_loss(P, levels=[0.2, 0.4, 0.5, 0.6, 0.8]):
  loss = 0.0
  T = len(P)
  if T == 0: return 0

  for p in levels:
    # w_i = np.sum([1 for p_t in P if p_t <= p])
    p_hat = np.sum([1 for p_t in P if p_t <= p]) / T
    loss += (p_hat-p)**2

  return loss