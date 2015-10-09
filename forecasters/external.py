#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.misc import logsumexp

from forecaster import Forecaster

class EWAForecaster(Forecaster):
  """Exponentialy weighted average forecaster"""
  def __init__(self, N, eta=None, w0=None):
    super(EWAForecaster, self).__init__(N)
    self.eta = eta
    if w0:
      self.w = w0
    else:
      self.w = np.ones(N,) / N
    self.t = 0

  def predict(self, e):
    return self.w.dot(e)

  def observe(self, l):
    self.t += 1
    if self.eta:
      eta_t = self.eta
    else:
      eta_t = 3*np.sqrt(2*np.log(self.N)/self.t)

    eta_t = 3*np.sqrt(2*np.log(self.N)/self.t)
    log_w = np.log(self.w)
    log_w_new = log_w - eta_t * l
    log_w_new = log_w_new - logsumexp(log_w_new)
    self.w = np.exp(log_w_new)
    assert self.w.shape == (self.N,)

  @property
  def weights(self):
      return self.w
  