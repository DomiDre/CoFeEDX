from xraydb import XrayDB
from thesis_utils.fileformats import load_xy
import numpy as np

def get_EI(datfile, min_E=None, max_E=None):
  E, I = load_xy(datfile, min_E, max_E, delimiter=',')
  sI = np.sqrt(I)
  intMax_I = max(I)
  I /= intMax_I
  sI /= intMax_I
  return E, I, sI

def get_CoFeRatio(elements, atomic_percent):
  idCo = elements.index('Co K')
  idFe = elements.index('Fe K')
  at_Co = float(atomic_percent[idCo])
  at_Fe = float(atomic_percent[idFe])
  sf_Co = at_Co/at_Fe
  return sf_Co

def get_x_ray_CoFe():
  xdb = XrayDB()
  lines_Fe = xdb.xray_lines('Fe', excitation_energy=20e3)
  lines_Co = xdb.xray_lines('Co', excitation_energy=20e3)
  intensity_Ka1 = (lines_Fe['Ka1'].intensity +
                  lines_Fe['Ka2'].intensity +
                  lines_Fe['Ka3'].intensity)
  return lines_Fe, lines_Co, intensity_Ka1

def get_x_ray_Fe():
  xdb = XrayDB()
  lines_Fe = xdb.xray_lines('Fe', excitation_energy=20e3)
  intensity_Ka1 = (lines_Fe['Ka1'].intensity +
                  lines_Fe['Ka2'].intensity +
                  lines_Fe['Ka3'].intensity)
  return lines_Fe, intensity_Ka1

def gaussian(x, mu, sig):
    return  np.exp(-((x-mu)/sig)**2/2)

def get_el_line(lines, E, min_E=0, sig=0.06):
  s = 0
  for line in lines:
    energy = lines[line].energy/1e3
    if energy < min_E:
      continue
    intensity = lines[line].intensity
    s += intensity*gaussian(E, energy, sig)
  return s
  # ax.plot(plot_E, s, color=color, label=label, marker='None')