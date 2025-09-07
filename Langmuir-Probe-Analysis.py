import numpy as np
from scipy.stats import linregress
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from scipy.interpolate import interp1d
from io import StringIO
import os
import glob

A = 2.2e-6  # enter probe tip area
folder = "/home/scien/Langmuir-Probe-Analysis/IEC-Langmuir-Ar"
file_list = glob.glob(os.path.join(folder, '*.lvm'))

e = 1.602e-19     # elementary charge (C)
m_e = 9.109e-31   # electron mass (kg)

# File selection menu
if file_list:
    print("Available files:")
    for i, file_path in enumerate(file_list):
        print(f"{i+1}: {os.path.basename(file_path)}")
    
    while True:
        try:
            choice = input(f"\nSelect file (1-{len(file_list)}): ")
            file_index = int(choice) - 1
            if 0 <= file_index < len(file_list):
                filepath = file_list[file_index]
                break
            else:
                print(f"Please enter a number between 1 and {len(file_list)}")
        except ValueError:
            print("Please enter a valid number")
    
    print(f'\n-------------- {os.path.basename(filepath)} --------------\n\n')

if os.path.exists(filepath):
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

    # (ion saturation region will be selected after floating potential is found)

    # interpolate data to find V_f at 0 current
    if np.any(I < 0) and np.any(I > 0):
        V_f = interp1d(I, V)(0) # create interpolated fcn f(I) = V, evaluate at 0
    else:
        V_f = np.nan

    # determine ion saturation region (default: V < V_f, fallback to V < 0)
    if np.isnan(V_f):
        ion_sat_mask = V < 0
    else:
        ion_sat_mask = V < V_f
    V_is = V[ion_sat_mask]
    I_is = I[ion_sat_mask]

    slope_is, intercept_is, R_is, _, _ = linregress(V_is, I_is)
    fit_is = slope_is * V_is + intercept_is
    R2 = R_is**2

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
            # fallback, use a few points around the floating potential
            center_idx = np.argmin(np.abs(V - V_f))
            start_idx = max(0, center_idx - 1)
            end_idx = min(len(V) - 1, center_idx + 1)
            V_er = V[start_idx:end_idx+1]
            I_er = I_e[start_idx:end_idx+1]
            # ensure positive currents for log operation
            if np.any(I_er <= 0):
                I_er = np.abs(I_er) + 1e-12

    slope_er, intercept_er, R_er, _, _ = linregress(V_er, np.log(I_er))

    if slope_er == 0:
        kT_e = np.nan
        print('warning: slope_er is zero, cannot compute kT_e')
    else:
        kT_e = 1.0 / slope_er

    # evaluate the extrapolated electron saturation current at the plasma potential V_p
    I_es = np.exp(slope_er * V_p + intercept_er)


    kT_e_J = kT_e * 1.60218e-19 # convert kT_e from eV to J

    n_e = (I_es / (e * A)) * np.sqrt((2 * np.pi * m_e) / (kT_e_J))

    # Debye length

    epsilon_0 = 8.854e-12  # vacuum permittivity (C^2 / N*m^2)
    k_B = 1.380649e-23     # Boltzmann (J/K)

    lambda_D = np.sqrt(epsilon_0 * kT_e_J / (n_e * e**2))

    # plot

    fig = plt.figure(figsize=(18, 12))
    
    # grid for plots, input panel, and results panel
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 0.8], height_ratios=[2, 2, 1], 
                         hspace=0.3, wspace=0.3)
    
    # 4 plots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # input parameters panel (top right)
    ax_params = fig.add_subplot(gs[0, 2])
    ax_params.axis('off')
    
    # results panel (bottom right)
    ax_results = fig.add_subplot(gs[1:, 2])
    ax_results.axis('off')

    ax1.plot(V, I)
    ax1.plot(V_is, fit_is, label='Ion Saturation Fit', color='red')
    ax1.axvline(V_f, color='purple', linestyle='--', label='V_f')
    ax1.axvline(V_p, color='green', linestyle='--', label='V_p')
    ax1.set_xlabel('Voltage (V)')
    ax1.set_ylabel('Current (A)')
    ax1.set_title(f'I-V Trace: {os.path.basename(filepath)}')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(V, I_e)
    ax2.axvline(V_f, color='purple', linestyle='--', label='V_f')
    ax2.axvline(V_p, color='green', linestyle='--', label='V_p')
    ax2.set_xlabel('Voltage (V)')
    ax2.set_ylabel('Electron Current (A)')
    ax2.set_title('Electron Current (I-V, Ion Fit Subtracted)')
    ax2.legend()
    ax2.grid(True)

    ax3.plot(V, -dIdV)
    ax3.axvline(V_f, color='purple', linestyle='--', label='V_f')
    ax3.axvline(V_p, color='green', linestyle='--', label='V_p')
    ax3.set_xlabel('Voltage (V)')
    ax3.set_ylabel('-dI/dV')
    ax3.set_title('Inverse Derivative of I-V Curve')
    ax3.legend()
    ax3.grid(True)

    pos_mask = I_e > 0  # prevents issues with log of neg values
    ax4.plot(V[pos_mask], np.log(I_e[pos_mask]))
    ax4.plot(V_er, slope_er * V_er + intercept_er, color='red', label='Electron Retardation Fit')
    ax4.axvline(V_f, color='purple', linestyle='--', label='V_f')
    ax4.axvline(V_p, color='green', linestyle='--', label='V_p')
    ax4.set_xlabel('Voltage (V)')
    ax4.set_ylabel('ln(I_e)')
    ax4.set_title('Electron Current Log Fit')
    ax4.legend()
    ax4.grid(True)

    # keep original computed values for reference
    A_orig = A
    V_f_orig = V_f
    V_p_orig = V_p
    V_er_start_orig = V_er[0] if len(V_er) > 0 else 0
    V_er_end_orig = V_er[-1] if len(V_er) > 0 else 0
    V_is_start_orig = V_is[0] if len(V_is) > 0 else V.min()
    V_is_end_orig = V_is[-1] if len(V_is) > 0 else 0

    # input boxes
    box_width = 0.06
    box_height = 0.03
    bbox = ax_params.get_position()
    left = bbox.x0
    right = bbox.x1
    top = bbox.y1
    # compute x position for input boxes slightly to the right of the param text area
    input_x = left + 0.6 * (right - left)
    input_w = 0.24 * (right - left)
    # vertical spacing for each line within the params box
    line_h = 0.12 * (bbox.y1 - bbox.y0)

    ax_A = plt.axes([input_x, top - line_h * 1.1, input_w, box_height])
    ax_Vf = plt.axes([input_x, top - line_h * 2.1, input_w, box_height])
    ax_Vp = plt.axes([input_x, top - line_h * 3.1, input_w, box_height])
    ax_Vis_start = plt.axes([input_x, top - line_h * 4.1, input_w, box_height])
    ax_Vis_end = plt.axes([input_x, top - line_h * 5.1, input_w, box_height])
    ax_Ver_start = plt.axes([input_x, top - line_h * 6.1, input_w, box_height])
    ax_Ver_end = plt.axes([input_x, top - line_h * 7.1, input_w, box_height])

    # textbox widgets with default values
    text_A = TextBox(ax_A, '', initial=f'{A:.1e}')
    text_Vf = TextBox(ax_Vf, '', initial=f'{V_f:.3f}')
    text_Vp = TextBox(ax_Vp, '', initial=f'{V_p:.3f}')
    text_Vis_start = TextBox(ax_Vis_start, '', initial=f'{V_is_start_orig:.3f}')
    text_Vis_end = TextBox(ax_Vis_end, '', initial=f'{V_is_end_orig:.3f}')
    text_Ver_start = TextBox(ax_Ver_start, '', initial=f'{V_er_start_orig:.3f}')
    text_Ver_end = TextBox(ax_Ver_end, '', initial=f'{V_er_end_orig:.3f}')

    # update plots when parameters change
    def update_plots(text):
        try:
            # get new values
            A_new = float(text_A.text)
            V_f_new = float(text_Vf.text)
            V_p_new = float(text_Vp.text)
            V_is_start_new = float(text_Vis_start.text)
            V_is_end_new = float(text_Vis_end.text)
            V_er_start_new = float(text_Ver_start.text)
            V_er_end_new = float(text_Ver_end.text)
            
            # recalculate ion fit with new range
            V_is_new = V[(V >= V_is_start_new) & (V <= V_is_end_new)]
            I_is_new = I[(V >= V_is_start_new) & (V <= V_is_end_new)]
            
            if len(I_is_new) > 1:
                slope_is_new, intercept_is_new, R_is_new, _, _ = linregress(V_is_new, I_is_new)
                fit_is_new = slope_is_new * V_is_new + intercept_is_new
                R2_new = R_is_new**2
                fit_is_full_new = slope_is_new * V + intercept_is_new
                I_e_new = I - fit_is_full_new
            else:
                # fallback to original values if invalid range
                slope_is_new, intercept_is_new, R2_new = slope_is, intercept_is, R2
                fit_is_new, I_e_new = fit_is, I_e
                V_is_new, I_is_new = V_is, I_is
            
            # recalculate electron fit with new parameters (allow fallback)
            V_er_new = V[(V >= V_er_start_new) & (V <= V_er_end_new) & (I_e_new > 0)]
            I_er_new = I_e_new[(V >= V_er_start_new) & (V <= V_er_end_new) & (I_e_new > 0)]

            # prepare fallback (original) electron-fit values
            slope_er_used = slope_er
            intercept_er_used = intercept_er
            kT_e_used = kT_e
            I_es_used = I_es
            kT_e_J_used = kT_e_J
            n_e_used = n_e
            lambda_D_used = lambda_D

            # attempt new electron fit; if it fails we keep the original values
            if len(I_er_new) > 1:
                try:
                    slope_er_new, intercept_er_new, _, _, _ = linregress(V_er_new, np.log(I_er_new))
                    slope_er_used = slope_er_new
                    intercept_er_used = intercept_er_new
                    kT_e_used = 1 / slope_er_used
                    I_es_used = np.exp(intercept_er_used)
                    kT_e_J_used = kT_e_used * 1.60218e-19
                    n_e_used = (I_es_used / (e * A_new)) * np.sqrt((2 * np.pi * m_e) / (kT_e_J_used))
                    lambda_D_used = np.sqrt(epsilon_0 * kT_e_J_used / (n_e_used * e**2))
                except Exception:
                    # keep original used values
                    pass

            # update plots (always redraw so changes to Vf/Vp or ion fit are shown)
            ax1.clear()
            ax1.plot(V, I)
            ax1.plot(V_is_new, fit_is_new, label='Ion Saturation Fit', color='red')
            ax1.axvline(V_f_new, color='purple', linestyle='--', label='V_f')
            ax1.axvline(V_p_new, color='green', linestyle='--', label='V_p')
            ax1.set_xlabel('Voltage (V)')
            ax1.set_ylabel('Current (A)')
            ax1.set_title(f'I-V Trace: {os.path.basename(filepath)}')
            ax1.legend()
            ax1.grid(True)

            ax2.clear()
            ax2.plot(V, I_e_new)
            ax2.axvline(V_f_new, color='purple', linestyle='--', label='V_f')
            ax2.axvline(V_p_new, color='green', linestyle='--', label='V_p')
            ax2.set_xlabel('Voltage (V)')
            ax2.set_ylabel('Electron Current (A)')
            ax2.set_title('Electron Current (I-V, Ion Fit Subtracted)')
            ax2.legend()
            ax2.grid(True)

            ax3.clear()
            ax3.plot(V, -dIdV)
            ax3.axvline(V_f_new, color='purple', linestyle='--', label='V_f')
            ax3.axvline(V_p_new, color='green', linestyle='--', label='V_p')
            ax3.set_xlabel('Voltage (V)')
            ax3.set_ylabel('-dI/dV')
            ax3.set_title('Inverse Derivative of I-V Curve')
            ax3.legend()
            ax3.grid(True)

            ax4.clear()
            pos_mask = I_e_new > 0
            ax4.plot(V[pos_mask], np.log(I_e_new[pos_mask]))
            # use the used slope/intercept for plotting the fit line
            ax4.plot(V_er_new if len(V_er_new)>0 else V_er, slope_er_used * (V_er_new if len(V_er_new)>0 else V_er) + intercept_er_used, color='red', label='Electron Retardation Fit')
            ax4.axvline(V_f_new, color='purple', linestyle='--', label='V_f')
            ax4.axvline(V_p_new, color='green', linestyle='--', label='V_p')
            ax4.set_xlabel('Voltage (V)')
            ax4.set_ylabel('ln(I_e)')
            ax4.set_title('Electron Current Log Fit')
            ax4.legend()
            ax4.grid(True)

            # update results
            ax_results.clear()
            ax_results.axis('off')
            results_text_new = f"""RESULTS

Ion saturation slope: {slope_is_new:.3e} A/V

R²: {R2_new:.4f}

Floating potential: {V_f_new:.3f} V

Plasma potential: {V_p_new:.3f} V

kT_e: {kT_e_used:.3f} eV

I_e,sat: {I_es_used:.3e} A

n_e: {n_e_used * 1e-6:.3e} cm⁻³

Debye length: {lambda_D_used * 1e3:.3f} mm
"""
            ax_results.text(0.05, 0.95, results_text_new, transform=ax_results.transAxes, 
                           fontsize=11, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

            plt.draw()
        except Exception as ex:
            print('update_plots error:', repr(ex))
    
    # apply button
    button_x = input_x
    button_y = top - line_h * 8.5
    button_w = input_w
    button_h = box_height * 1.2
    ax_button = plt.axes([button_x, button_y, button_w, button_h])
    apply_button = Button(ax_button, 'Apply')
    apply_button.on_clicked(lambda x: update_plots(None))

    # parameter labels
    params_text = f"""PARAMETERS

Probe area A:                         (default: {A_orig:.1e} m²)

Floating potential V_f:               (default: {V_f_orig:.3f} V)

Plasma potential V_p:                 (default: {V_p_orig:.3f} V)

Ion sat start:                        (default: {V_is_start_orig:.3f} V)

Ion sat end:                          (default: {V_is_end_orig:.3f} V)

Electron ret start:                   (default: {V_er_start_orig:.3f} V)

Electron ret end:                     (default: {V_er_end_orig:.3f} V)
"""
    
    ax_params.text(0.05, 0.95, params_text, transform=ax_params.transAxes, 
                  fontsize=9, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    # results panel
    results_text = f"""RESULTS

Ion saturation slope: {slope_is:.3e} A/V

R²: {R2:.4f}

Floating potential: {V_f:.3f} V

Plasma potential: {V_p:.3f} V

kT_e: {kT_e:.3f} eV

I_e,sat: {I_es:.3e} A

n_e: {n_e * 1e-6:.3e} cm⁻³

Debye length: {lambda_D * 1e3:.3f} mm
"""
    
    ax_results.text(0.05, 0.95, results_text, transform=ax_results.transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
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
    plt.show()
else:
    print("No .lvm files found in the folder.")