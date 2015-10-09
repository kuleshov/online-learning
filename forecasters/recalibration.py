#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from forecaster import Forecaster
from calibration import EWACalibratedForecaster

class EWARecalibratedForecaster(Forecaster):
  """Recalibrates online probability estimates"""
  def __init__(self, N, eta=None):
    super(EWARecalibratedForecaster, self).__init__(N+1)
    self.last_i = None
    # F_cal[i] is responsible for iterval [i/N, (i+1)/N]
    if (isinstance(eta, np.ndarray) or isinstance(eta, list)) and len(eta) == N:
      self.F_cal = [EWACalibratedForecaster(N, eta[i]) for i in range(N)]
    else:
      self.F_cal = [EWACalibratedForecaster(N, eta) for i in range(N)]

  def predict(self, p_raw):
    if p_raw == 1.0:
      i = self.N-2
    else:
      i = int(np.floor(p_raw*(self.N-1)))
    self.last_i = i
    return self.F_cal[i].predict()

  def expected_prediction(self, p_raw):
    if p_raw == 1.0:
      i = self.N-2
    else:
      i = int(np.floor(p_raw*(self.N-1)))
    self.last_i = i
    return self.F_cal[i].expected_prediction()

  def observe(self, y_t):
    if self.last_i is None:
      raise ValueError("Need to observe an uncalibrated p first")

    i = self.last_i
    self.F_cal[i].observe(y_t)
    self.last_i = None

  @property
  def examples_seen(self):
      return [F.examples_seen for F in self.F_cal]
  
      
