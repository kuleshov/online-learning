#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.misc import logsumexp

from forecaster import Forecaster
from internal import EWAInternalForecaster

class EWACalibratedForecaster(Forecaster):
  """Produces calibrated estimates by minimizng internal regret"""
  def __init__(self, N, eta):
    super(EWACalibratedForecaster, self).__init__(N+1)
    self.F_int = EWAInternalForecaster(N+1, eta)
    self.e = np.array([float(i)/N for i in xrange(N+1)])

  def predict(self):
    w = self.F_int.weights.reshape(self.N,)
    ind = np.random.multinomial(1, w)
    return ind.dot(self.e)

  def expected_prediction(self):
    w = self.F_int.weights.reshape(self.N,)
    return w.T.dot(self.e)

  def observe(self, y_t):
    l = np.array([y_t - e for e in self.e])
    self.F_int.observe(l)

  @property
  def weights(self):
    return self.F_int.weights

# ----------------------------------------------------------------------------
# Helpers

def calib_loss(Y, P, N):
  loss = 0.0
  T = len(Y)
  if T == 0: return 0

  for i in xrange(N+1):
    p = float(i)/N
    w_i = np.sum([p_t for p_t in P if p_t == p])
    y_i = np.sum([p_t*y_t for p_t, y_t in zip(P,Y) if p_t == p])
    rho_i = y_i / w_i if w_i else 0.0
    loss += w_i*(rho_i-p)**2

  return loss / T
