#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.misc import logsumexp

from forecaster import Forecaster
from external import EWAForecaster

class EWAInternalForecaster(Forecaster):
  """Minimizes internal regret via reduction to external regret"""
  def __init__(self, N, eta, w0=None):
    super(EWAInternalForecaster, self).__init__(N)
    self.F_ext = EWAForecaster(N*(N-1), eta)
    if w0:
      self.w = w0.reshape((N,1))
    else:
      self.w = np.ones((N,1)) / N

  def predict(self, e):
    e = e.reshape((self.N,1))
    w = self.w.reshape((self.N,1))
    return w.T.dot(e)

  def observe(self, l):
    N = self.N
    M = N*(N-1)
    p = self.w.reshape((N,1))
    p_ext = np.tile(p, (1, M))
    l_int = l.reshape(1,N)

    P_matrices = list()
    ij_indices = self._get_ij_ind()
    for k, (i,j) in enumerate(ij_indices):
      p_ext[j][k] += p_ext[i][k]
      p_ext[i][k] = 0
      
      # make permutation matrix
      P = np.eye(N)
      P[i][i] = 0
      P[j][i] = 1
      v1 = p_ext[:,k].reshape(N,)
      v2 = P.dot(p).reshape(N,)
      assert np.linalg.norm(v1 - v2) < 1e-4
      P_matrices.append(P)

      l_ext = l_int.dot(p_ext).reshape(M,)

    self.F_ext.observe(l_ext)

    ij_indices = self._get_ij_ind()
    Psum = np.zeros((N,N))
    for (d,P) in zip(self.F_ext.weights, P_matrices):
      Psum += d*P

    lam, Q = np.linalg.eig(Psum)
    fixed_pts = [Q[:,i] for i in xrange(N) if np.abs(lam[i]-1) < 1e-3]
    assert len(fixed_pts) == 1

    self.w = np.abs(fixed_pts[0].reshape(N,1))

  @property
  def weights(self):
      return self.w
  

  def _get_ij_ind(self):
    return ((i,j) for i in xrange(self.N) for j in xrange(self.N) if i != j)
