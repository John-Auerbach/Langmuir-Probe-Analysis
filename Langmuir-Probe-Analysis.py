import numpy as np
from scipy.stats import linregress
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from io import StringIO
import os
import glob

A = 1e-5  # enter probe tip area
folder = 'IEC Langmuir'
file_list = glob.glob(os.path.join(folder, '*.lvm'))

e = 1.602e-19     # elementary charge (C)
m_e = 9.109e-31   # electron mass (kg)

# Process only the first file
if file_list:
    filepath = file_list[0]
    print(f'\n-------------- {os.path.basename(filepath)} --------------\n\n')

    with open(filepath, 'r') as f:
        lines = f.readlines()

    header_end_indices = [i for i, line in enumerate(lines) if '***End_of_Header***' in line]

    data_start_idx = header_end_indices[1] + 1 # data starts after the second header end
    data_str = ''.join(lines[data_start_idx:])
    data = pd.read_csv(StringIO(data_str), sep='\t', comment='X', skipinitialspace=True)
    
    # drop the comment column
    data = data.drop(columns=[col for col in data.columns if 'Comment' in col or data[col].isna().all()])


    # set up LP trace data
    V = data.iloc[:, 0].values
    I = data.iloc[:, 1].values

    ion_sat_mask = V < 0
    V_is = V[ion_sat_mask]
    I_is = I[ion_sat_mask]

    slope_is, intercept_is, R_is, _, _ = linregress(V_is, I_is)
    fit_is = slope_is * V_is + intercept_is
    R2 = R_is**2

    # interpolate data to find V_f at 0 current
    if np.any(I < 0) and np.any(I > 0):
        V_f = interp1d(I, V)(0) # create interpolated fcn f(I) = V, evaluate at 0
    else:
        V_f = np.nan

    # subtract ion fit

    fit_is_full = slope_is * V + intercept_is # extend ion sat fit to all voltages not just masked region
    I_e = I - fit_is_full

    # find plasma potential

    dIdV = np.gradient(I, V)
    V_p = V[np.argmin(- dIdV)]

    # find T_e and n_e

    elec_ret_mask = (V > V_f) & (V < V_p) & (V > 0) & (I_e > 0) # last two conditions allow log operations to ignore negatives
    V_er = V[elec_ret_mask]
    I_er = I_e[elec_ret_mask]

    if len(I_er) < 2: # if less than 2 points to do lin regression in that region, just choose surrounding points
        idx_lower_candidates = np.where((V <= V_f))[0]
        idx_upper_candidates = np.where((V >= V_p))[0]
        
        if len(idx_lower_candidates) > 0 and len(idx_upper_candidates) > 0:
            idx_lower = np.max(idx_lower_candidates)
            idx_upper = np.min(idx_upper_candidates)
            V_er = V[[idx_lower, idx_upper]]
            I_er = I_e[[idx_lower, idx_upper]]
        else:
            # Fallback: use a few points around the floating potential
            center_idx = np.argmin(np.abs(V - V_f))
            start_idx = max(0, center_idx - 1)
            end_idx = min(len(V) - 1, center_idx + 1)
            V_er = V[start_idx:end_idx+1]
            I_er = I_e[start_idx:end_idx+1]
            # Ensure we have positive currents for log operation
            if np.any(I_er <= 0):
                I_er = np.abs(I_er) + 1e-12

    slope_er, intercept_er, R_er, _, _ = linregress(V_er, np.log(I_er))

    kT_e = 1 / slope_er
    I_es = np.exp(intercept_er)

    kT_e_J = kT_e * 1.60218e-19 # convert kT_e from eV to J

    n_e = (I_es / (e * A)) * np.sqrt((2 * np.pi * m_e) / (kT_e_J))

    # Debye length

    epsilon_0 = 8.854e-12  # vacuum permittivity (C^2 / N*m^2)
    k_B = 1.380649e-23     # Boltzmann (J/K)

    lambda_D = np.sqrt(epsilon_0 * kT_e_J / (n_e * e**2))

    # plot

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].plot(V, I)
    axs[0, 0].plot(V_is, fit_is, label='Ion Saturation Fit', color='red')
    axs[0, 0].axvline(V_f, color='purple', linestyle='--', label='V_f')
    axs[0, 0].axvline(V_p, color='green', linestyle='--', label='V_p')
    axs[0, 0].set_xlabel('Voltage (V)')
    axs[0, 0].set_ylabel('Current (A)')
    axs[0, 0].set_title(f'I-V Trace: {os.path.basename(filepath)}')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(V, I_e)
    axs[0, 1].axvline(V_f, color='purple', linestyle='--', label='V_f')
    axs[0, 1].axvline(V_p, color='green', linestyle='--', label='V_p')
    axs[0, 1].set_xlabel('Voltage (V)')
    axs[0, 1].set_ylabel('Electron Current (A)')
    axs[0, 1].set_title('Electron Current (I-V, Ion Fit Subtracted)')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[1, 0].plot(V, -dIdV)
    axs[1, 0].axvline(V_f, color='purple', linestyle='--', label='V_f')
    axs[1, 0].axvline(V_p, color='green', linestyle='--', label='V_p')
    axs[1, 0].set_xlabel('Voltage (V)')
    axs[1, 0].set_ylabel('-dI/dV')
    axs[1, 0].set_title('Inverse Derivative of I-V Curve')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    pos_mask = I_e > 0  # prevents issues with log of neg values
    axs[1, 1].plot(V[pos_mask], np.log(I_e[pos_mask]))
    axs[1, 1].plot(V_er, slope_er * V_er + intercept_er, color='red', label='Electron Retardation Fit')
    axs[1, 1].axvline(V_f, color='purple', linestyle='--', label='V_f')
    axs[1, 1].axvline(V_p, color='green', linestyle='--', label='V_p')
    axs[1, 1].set_xlim(left=0)
    axs[1, 1].set_xlabel('Voltage (V)')
    axs[1, 1].set_ylabel('ln(I_e)')
    axs[1, 1].set_title('Electron Current Log Fit')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    
    print('\n\n')
    print('Ion saturation region slope =', slope_is, 'A/V')
    print('R² =', R2)
    print('Floating potential =', V_f, 'V')
    print('Plasma potential =', V_p, 'V')
    print('kT_e =', kT_e, 'eV')
    print('I_e,sat =', I_es, 'A')
    print('n_e =', n_e * 1e-6, 'cm^-3')
    print('Debye length =', lambda_D * 1e3, 'mm')
    print('\n\n')
    
    print("Displaying plots... Close the plot window to continue or press Ctrl+C to exit.")
    plt.show()  # Blocking show - keeps plot open until closed
else:
    print("No .lvm files found in the folder.")