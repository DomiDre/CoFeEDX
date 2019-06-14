from xraydb import XrayDB
import numpy as np
from bs4 import BeautifulSoup

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

def load_xy(xyfile, x_min=None, x_max=None, delimiter=None):
  data_array = np.genfromtxt(xyfile, delimiter=delimiter)
  x = data_array[:, 0]
  y = data_array[:, 1]
  valid_values = np.ones(len(x), dtype=bool)
  if x_min:
    valid_values = np.logical_and(valid_values, x > x_min)
  if x_max:
    valid_values = np.logical_and(valid_values, x < x_max)
  x = x[valid_values]
  y = y[valid_values]
  return x, y

def load_edx_htm(htmfile):
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

def init_fig():
  fig = plt.figure()
  left, bottom = 0.19, 0.15
  ax = fig.add_axes([left,bottom, 1-left-0.01, 1-bottom-0.01])
  return fig, ax

def setup_ax(ax, min_E, max_E, min_I, max_I):
  ax.set_xlabel('$\mathit{E} \, / \, keV$')
  ax.set_ylabel('$\mathit{I} \, / \, a.u.$')
  ax.set_xlim([min_E, max_E])
  ax.set_ylim([min_I, max_I])
  ax.legend(loc='upper right')

def plot_model(ax, E, I, label=None):
  ax.plot(E, I, label=label, color='black', marker='None', linestyle='-', alpha=0.5)

def errorbar(ax, E, I, sI, label=None):
  ax.errorbar(E, I, sI, label=label, marker='None',\
              linestyle='None', capsize=0, color='#ca0020')

def plot_ratio(ax, ratio_FeCo):
  ax.plot([],[],label='Co:Fe 1:'+str(ratio_FeCo), marker='None', ls='None')

def plot_lines(ax, min_E, plot_E, lines, color, label=None, sf=1):
  def gaussian(x, mu, sig):
    return  np.exp(-((x-mu)/sig)**2/2)
  s = 0
  for line in lines:
    energy = lines[line].energy/1e3
    if energy < min_E:
      continue
    intensity = lines[line].intensity*sf
    s += intensity*gaussian(plot_E, energy, 0.06)
  return s
