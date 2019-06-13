import sys, os
import matplotlib.pyplot as plt
import numpy as np
from math import floor, log10

# remove some annoying warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

from EDX.edx import *

# change the next two names
savefile = 'EDX_sample.png'
data_name = 'samplename'

min_E, max_E = 5.5, 8.5
min_I, max_I = 0.,1.3
lines_Fe, lines_Co, intensity_Ka1 = get_x_ray_CoFe()
num_measurements = 5

current_sample = data_name+'_1'
datfile = f'./{current_sample}.txt'
E, I, sI = get_EI(datfile, min_E)
num_pts = len(E)
energies = np.zeros((num_pts, num_measurements))
intensities = np.zeros((num_pts, num_measurements))
sig_intensities = np.zeros((num_pts, num_measurements))
ratios = []

for i in range(1, 1+num_measurements):
  current_sample = data_name+'_'+str(i)
  datfile = f'./{current_sample}.txt'

  htmfile = datfile.replace('.txt', '.htm')
  E, I, sI = get_EI(datfile, min_E)

  energies[:, i-1] = E
  intensities[:, i-1] = I
  sig_intensities[:, i-1] = sI

  elements, weight_percent, atomic_percent = load_edx_htm(htmfile)
  sf_Co = get_CoFeRatio(elements, atomic_percent)
  print('Co:Fe = ', 1/sf_Co)
  ratios.append(1/sf_Co)

mE = np.mean(energies, axis=1)
mI = np.mean(intensities, axis=1)
msI = np.sqrt(np.mean(sig_intensities**2, axis=1))

mRatio = np.mean(ratios)
sRatio = np.std(ratios, ddof=1)
ratio_FeCo = np.round(mRatio, 2)
sig_ratio_FeCo = int(np.round(10*np.round(sRatio,3) / (10**floor(log10(np.round(sRatio,1))))))

Fe_line = get_el_line(lines_Fe, E, min_E)
Co_line = get_el_line(lines_Co, E, min_E)
model = Fe_line/intensity_Ka1 + Co_line/mRatio


# Plot
left, bottom = 0.16, 0.16
fig = plt.figure()
ax = fig.add_axes([left,bottom, 1-left-0.01, 1-bottom-0.01])

ax.errorbar(mE, mI, msI, ls="None", zorder=0)
ax.plot(E, model, color='black', marker="None", zorder=1, alpha=0.5)
ax.set_xlim(min_E, max_E)
ax.set_ylim(0, 1.1)
ax.text(0.96, 0.96, f'{data_name}\n'+'$\mathit{n}_{Fe}/\mathit{n}_{Co}$' + f' = {ratio_FeCo}({sig_ratio_FeCo})',
  horizontalAlignment='right',
  verticalAlignment='top',
  transform=ax.transAxes)
ax.set_xlabel('$\mathit{E} \, / \, keV$')
ax.set_ylabel('$\mathit{I} \, / \, a.u.$')
fig.savefig('./' + savefile)

print('Co:Fe = ' + str(np.mean(ratios)) + ' +/- ' + str(np.std(ratios,ddof=1)))