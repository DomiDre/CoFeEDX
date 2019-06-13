import numpy as np
from xraydb import XrayDB
from bs4 import BeautifulSoup

import lmfit

class EDXFit():
  def __init__(self):
    self.E = None
    self.I = None
    self.sI = None
    self.min_E = 0
    self.max_E = np.inf

    self.xdb = XrayDB()
    self.lines_Fe = self.xdb.xray_lines('Fe', excitation_energy=20e3)
    self.lines_Co = self.xdb.xray_lines('Co', excitation_energy=20e3)
    self.intensity_Ka1 = (self.lines_Fe['Ka1'].intensity +
                    self.lines_Fe['Ka2'].intensity +
                    self.lines_Fe['Ka3'].intensity)

  def load_txt_file(self, txtfile, E_min=None, E_max=None, delimiter=','):
    data_array = np.genfromtxt(txtfile, delimiter=delimiter)
    E = data_array[:, 0]
    I = data_array[:, 1]

    # sort in ascending order
    E, I = zip(*sorted(zip(E, I)))
    E = np.asarray(E)
    I = np.asarray(I)

    # cut off data points out of range
    valid_values = np.ones(len(I), dtype=bool)
    if E_min:
      valid_values = np.logical_and(valid_values, E > E_min)
    if E_max:
      valid_values = np.logical_and(valid_values, E < E_max)
    E = E[valid_values]
    I = I[valid_values]

    # Poisson Error
    sI = np.sqrt(I)

    # scale to max. value
    intMax_I = max(I)
    I /= intMax_I
    sI /= intMax_I
    self.E = E
    self.I = I
    self.sI = sI
    return E, I, sI

  def load_edx_htm(self, htmfile):
    elements = []
    weight_percent = []
    atomic_percent = []
    data_array = [elements,weight_percent,atomic_percent]
    with open(htmfile, 'r', errors='ignore') as f:
      soup = BeautifulSoup(f, "lxml")
      table_rows = soup.select("table tr")
      for row in table_rows:
        columns = row.findAll('td')
        for i, row in enumerate(columns):
          if i>2: continue
          if row.contents:
            el = row.contents[0]
            data_array[i].append(el)
    return data_array

  def estimateBackground(self, min_E_bg, max_E_bg, subtract=True):
    assert(self.E is not None)
    bg_E_range = np.logical_and(self.E > min_E_bg, self.E < max_E_bg)

    E_bg = self.E[bg_E_range]
    I_bg = self.I[bg_E_range]
    sI_bg = self.sI[bg_E_range]

    self.bg = np.mean(I_bg)
    self.sbg = np.std(I_bg, ddof=1)
    print(f'Estimated Background to: {self.bg} +/- {self.sbg}')

    self.I -= self.bg
    self.sI = np.sqrt(self.sI**2 + self.sbg**2)

  def gaussian(self, x, mu, sig):
    return  np.exp(-((x-mu)/sig)**2/2)

  def residuum(self, p, E, I, sI, Imodel):
    return (I - Imodel(p, E))/sI

  def CoFe_model(self, p, E):
    bg = p['bg']
    sigma = p['sigma']
    r_Fe = p['r_Fe']
    r_Co = p['r_Co']
    model = np.zeros(len(E))
    for line in self.lines_Fe:
      energy = self.lines_Fe[line].energy/1e3
      if energy < self.min_E or energy > self.max_E:
        continue
      intensity = self.lines_Fe[line].intensity
      model += r_Fe*intensity*self.gaussian(E, energy, sigma)

    for line in self.lines_Co:
      energy = self.lines_Co[line].energy/1e3
      if energy < self.min_E or energy > self.max_E:
        continue
      intensity = self.lines_Co[line].intensity
      model += r_Co*intensity*self.gaussian(E, energy, sigma)
    return model + bg

  def init_params_CoFe(self):
    p = lmfit.Parameters()
    p.add('bg', 0.01, min=0, vary=True)
    p.add('sigma', 0.06, vary=True)
    p.add('r_Co', 1, vary=True)
    p.add('r_Fe', 2, vary=True)
    self.p = p
    return p
  
  def fit(self, p, Imodel):
    fit_result = lmfit.minimize(self.residuum, p, args=(self.E, self.I, self.sI, Imodel))
    print(lmfit.fit_report(fit_result))
    self.p = fit_result.params