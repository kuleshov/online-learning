#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from forecaster import Forecaster
from calibration import EWACalibratedForecaster, RunningAverageForecaster

class EWARecalibratedForecaster(Forecaster):
  """Recalibrates online probability estimates"""
  def __init__(self, N, eta=None):
    super(EWARecalibratedForecaster, self).__init__(N+1)
    # F_cal[i] is responsible for iterval [i/N, (i+1)/N]
    if (isinstance(eta, np.ndarray) or isinstance(eta, list)) and len(eta) == N:
      self.F_cal = [EWACalibratedForecaster(N, eta[i]) for i in range(N)]
    else:
      self.F_cal = [EWACalibratedForecaster(N, eta) for i in range(N)]

  def predict(self, p_raw):
    i = self._get_idx(p_raw)
    out = self.F_cal[i].predict()
    return out

  def expected_prediction(self, p_raw):
    i = self._get_idx(p_raw)
    return self.F_cal[i].expected_prediction()

  def observe(self, y_t, p_raw):
    i = self._get_idx(p_raw)
    self.F_cal[i].observe(y_t)

  def _get_idx(self, p_raw):
    if p_raw == 1.0:
      i = self.N-2
    else:
      i = int(np.floor(p_raw*(self.N-1)))
    return i

  @property
  def examples_seen(self):
      return [F.examples_seen for F in self.F_cal]
      

class EWARecalibratedRegressionForecaster(EWARecalibratedForecaster):
  def __init__(self, N, eta=None):
    super(EWARecalibratedRegressionForecaster, self).__init__(N, eta)
    # self.F_cal = [RunningAverageForecaster(N) for i in range(N)]

  def observe(self, y_t, p_raw):
    for i in range(len(self.F_cal)):
      o_t = 1 if p_raw <= (float(i) / self.N) else 0
      self.F_cal[i].observe(o_t) 