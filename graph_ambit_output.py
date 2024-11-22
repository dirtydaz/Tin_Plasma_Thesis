import re
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import matplotlib.ticker as ticker

def parse_output_o23(output, y='a', width=[60,120], T=26):
    with open(output, 'r') as file:
        lines = file.readlines()
        content = file.read()

    energy_levels = []
    current_J = None
    current_P = None

    for line in lines:
        # Match the header lines to capture J and P
        header_match = re.match(r'Solutions for J = (\d+(?:\.\d+)?), P = (even|odd)', line)
        if header_match:
            current_J = float((header_match.group(1)))
            current_P = header_match.group(2)
            continue

        # Match the energy levels
        energy_match = re.match(r'(\d+):\s*(-?\d+\.\d+|-?\d+)', line)
        if energy_match:
            index = int(energy_match.group(1))
            energy = float(energy_match.group(2))
            energy_levels.append([int(current_J * 2), current_P, index, energy])

    # Create a dictionary mapping state labels to (energy, J)
    state_info_dict = {}
    for j, parity, i, E in energy_levels:
        state_label = f"{j}{parity[0]}:{i}"
        state_info_dict[state_label] = (E, j)

    transition_pattern = re.compile(r'(\S+:\d+) -> (\S+:\d+) = ([\deE.\-+]+)')

    transitions = []
    transition_count = 0
    collect_transitions = False

    # Loop through each line and search for transitions
    for line in lines:
        # Check for the mention of E1 transitions
        if "E1 transition strengths (S):" in line:
            transition_count += 1

            # Start collecting transitions after the second mention
            if transition_count == 2:
                collect_transitions = True
            # Stop collecting after the third mention
            elif transition_count == 3:
                collect_transitions = False
                break

        # Collect transitions if we're between the second and third mentions
        if collect_transitions:
            match = transition_pattern.search(line)
            if match:
                initial = match.group(1)
                final = match.group(2)
                weight = float(match.group(3))
                transitions.append([initial, final, weight])

    transition_data = []

    atomic_units = 4.3597447222071e-18
    e = 1.602e-19

    if y == 's':
        # Iterate over the transitions and calculate the transition energies
        for initial, final, weight in transitions:
            initial_energy = state_info_dict[initial][0]
            final_energy = state_info_dict[final][0]
            transition_energy = abs(final_energy - initial_energy) * atomic_units / e
            transition_data.append((initial, final, float(weight), transition_energy))
    elif y == 'a':
        for initial, final, weight in transitions:
            initial_energy, initial_2j = state_info_dict[initial]
            final_energy, final_2j = state_info_dict[final]
            transition_energy = abs(final_energy - initial_energy) * atomic_units / e
            # Determine which state has higher energy
            if initial_energy > final_energy:
                higher_2j = initial_2j
            else:
                higher_2j = final_2j
            # Divide weight by higher_2j
            new_weight = 1.0608*(10**6)*transition_energy**3*float(weight)/(higher_2j+1)
            transition_data.append((initial, final, new_weight, transition_energy))
            
    elif y == 'i':
        Z = 0
        for initial, final, weight in transitions:
            initial_energy, initial_2j = state_info_dict[initial]
            final_energy, final_2j = state_info_dict[final]
            if initial_energy >= final_energy:
                Z += np.exp(-initial_energy* atomic_units/(T*e))
            else:
                Z += np.exp(-final_energy* atomic_units/(T*e))
                
        for initial, final, weight in transitions:
            initial_energy, initial_2j = state_info_dict[initial]
            final_energy, final_2j = state_info_dict[final]
            transition_energy = abs(final_energy - initial_energy) * atomic_units / e
             # Determine which state has higher energy
            if initial_energy > final_energy:
                higher_2j = initial_2j
                higher_E = initial_energy
            else:
                higher_2j = final_2j   
                higher_E = final_energy
            g = higher_2j+1
            einstein_weight = 1.0608*(10**6)*transition_energy**3*float(weight)/g
            intensity_weight = np.exp(-higher_E* atomic_units/(T*e))*g*einstein_weight*transition_energy/(4*np.pi*Z)
            transition_data.append((initial, final, intensity_weight, transition_energy))
        
    else:
        # Handle unexpected y value
        raise ValueError("Invalid value for y. Expected 's' or 'a'.")

    print(len(transitions), len(energy_levels), len(state_info_dict))

    # Extract weights and energies
    weights = np.array([entry[2] for entry in transition_data])
    energies = np.array([entry[3] for entry in transition_data])

    # Normalize the weights
    # normalized_weights = weights / np.sum(weights)

    # Define Lorentzian function
    def lorentzian(x, x0, gamma):
        return gamma / (np.pi * ((x - x0) ** 2 + gamma ** 2))

    # Create x values for the Lorentzian plot
    gamma = 0.5  # Width of the Lorentzian peak
    x = np.linspace(width[0], width[1], 1000)

    # Calculate the total Lorentzian distribution
    total_lorentzian = np.zeros_like(x)
    for energy, weight in zip(energies, weights):
        total_lorentzian += lorentzian(x, energy, gamma) * weight

    return energies, weights, total_lorentzian, x



def process_output(idx, output, y, width):
    energies, weights, total_lorentzian, x = parse_output_o23(output, y, width)
    return {
        'idx': idx,
        'energies': energies,
        'weights': weights,
        'total_lorentzian': total_lorentzian,
        'x': x
    }

def plot_spectra(outputs, fig_name='', band=False, plot_choice='both', y='a', width=[60, 110],stems=0.1,exp=0):
    colors = ['black', 'red', 'blue', 'green', 'brown']
    plt.figure(figsize=(6, 4))

    # Parallelize data preparation
    results = Parallel(n_jobs=-1)(
        delayed(process_output)(idx, output, y, width) for idx, output in enumerate(outputs)
    )

    # Now plot the results sequentially
    for result in results:
        idx = result['idx']
        energies = result['energies']
        weights = result['weights']
        total_lorentzian = result['total_lorentzian']
        x = result['x']

        # Plot the Lorentzian distribution
        plt.plot(x, total_lorentzian, label=f'Total Lorentzian {idx}', color=colors[idx])

        if plot_choice == 'both':
            # Select the top x% of weights
            num_top = max(1, int(stems * len(weights)))  # Ensure at least one point is selected
            sorted_indices = np.argsort(weights)[::-1]  # Indices that would sort weights in descending order
            top_indices = sorted_indices[:num_top]
            top_energies = energies[top_indices]
            top_weights = weights[top_indices]

            # Plot the stem plot (discrete points) for the top 10% weights
            markerline, stemlines, baseline = plt.stem(
                top_energies, top_weights, basefmt=" ",
                label=f'Discrete Weights {idx}', linefmt=colors[idx], markerfmt=colors[idx]
            )
            plt.setp(markerline, markersize=4)
            plt.setp(stemlines, linewidth=1)

    # Rest of your plotting code remains the same
    plt.xlabel('Energy (eV)')
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((exp, exp))  # Forces scientific notation
    plt.gca().yaxis.set_major_formatter(formatter)
    if y == 'a':
        plt.ylabel("A s$^{-1}$")
    elif y == 's':
        plt.ylabel('S (a.u.)')
    elif y == 'i':
        plt.ylabel('I (eV s$^{-1}$ atom$^{-1}$)')
    plt.xlim(width[0], width[1])
    plt.ylim(bottom=0)
    plt.gca().xaxis.set_minor_locator(MultipleLocator(2))
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().tick_params(axis='x', which='minor', top=True, direction='in')
    plt.gca().tick_params(axis='x', which='major', top=True, direction='in')
    plt.gca().tick_params(axis='x', direction='inout')
    plt.gca().tick_params(axis='x', which='minor', direction='inout')
    plt.gca().tick_params(axis='y', which='major', direction='in')
    plt.gca().tick_params(axis='y', which='minor', direction='in')

    if band:
        plt.axvspan(90.92, 92.75, color='gray', alpha=0.3)

    if len(fig_name) != 0:
        plt.savefig(f"figures/{fig_name}.pdf")

    plt.tight_layout()
    plt.show()
    return
