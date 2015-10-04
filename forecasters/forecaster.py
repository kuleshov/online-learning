#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Forecaster(object):
  """Base forecaster class"""
  def __init__(self, N):
    super(Forecaster, self).__init__()
    self.N = N
  
  def predict(e):
    raise NotImplementedError("Function must be extended by child class")

  def observe(l):
    raise NotImplementedError("Function must be extended by child class")
