import matplotlib.pyplot as plt
import re
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def parse_CI_solutions(output):
    """
    Parses the output file to extract energy levels, states, and their associated configurations with weight percentages.
    :param output: Path to the output file.
    :return: List of dictionaries, each containing J, Parity, index, energy, and configurations with their weight percentages.
    """
    import re

    with open(output, 'r') as file:
        lines = file.readlines()

    energy_levels = []
    current_J = None
    current_P = None

    for line in lines:
        # Match the header lines to capture J and P
        header_match = re.match(r'Solutions for J = (\d+(?:\.\d+)?), P = (even|odd)', line)
        if header_match:
            current_J = float(header_match.group(1))
            current_P = header_match.group(2)
            continue

        # Match the energy levels
        energy_match = re.match(r'(\d+):\s*(-?\d+\.\d+)', line)
        if energy_match:
            index = int(energy_match.group(1))
            energy = float(energy_match.group(2))
            configurations = []

            # Collect configurations and their weights
            config_index = lines.index(line) + 1
            while config_index < len(lines) and re.match(r'\s+\S+.*\s+\d+\.\d+%', lines[config_index]):
                config_line = lines[config_index].strip()
                config_match = re.match(r'(\S+(?:\s+\S+)*?)\s+(\d+\.\d+)%', config_line)
                if config_match:
                    configuration = config_match.group(1)
                    weight = float(config_match.group(2))
                    configurations.append((configuration, weight))
                config_index += 1

            # Append the energy level data with configurations
            energy_levels.append({
                'J': int(current_J), 
                'P': current_P, 
                'index': index, 
                'energy': energy, 
                'configurations': configurations
            })

    return energy_levels

def filter_energies(energy_levels, J, P):
    """
    Filters energy levels by specified J and P.
    """
    return [entry['energy'] for entry in energy_levels if entry['J'] == J and entry['P'] == P]


def save_combined_parity_plots(files, even_pdf="even_parity_plots.pdf", odd_pdf="odd_parity_plots.pdf"):
    """
    Parses multiple files, determines all J and P combinations, and saves the top 4 even and 4 odd
    combinations with the highest number of states to two separate figures, saved as PDFs.
    
    :param files: List of paths to the output files.
    :param even_pdf: Filename for saving the combined even parity plots.
    :param odd_pdf: Filename for saving the combined odd parity plots.
    """
    # Dictionary to count the number of states for each J and P combination
    combination_counts = {}

    # Parse all files to count the number of states for each combination
    for file in files:
        energy_levels = parse_CI_solutions(file)
        for level in energy_levels:
            J, P = level['J'], level['P']  # Assuming 'J' and 'P' keys exist in the parsed data
            key = (J, P)
            combination_counts[key] = combination_counts.get(key, 0) + 1

    # Separate even and odd combinations
    even_combinations = [(key, count) for key, count in combination_counts.items() if key[1] == 'even']
    odd_combinations = [(key, count) for key, count in combination_counts.items() if key[1] == 'odd']

    # Sort by count in descending order and take the top 4 for each parity
    top_even = sorted(even_combinations, key=lambda x: x[1], reverse=True)[:4]
    top_odd = sorted(odd_combinations, key=lambda x: x[1], reverse=True)[:4]

    # Color mapping for specific labels
    color_map = {
        "CI": "black",
        "NOCI": "red",
        "RMT": "blue",
        "odRMT": "green",
    }

    # Function to create and save combined plots for a given parity
    def save_combined_plot(pdf_filename, parity_name, combinations):
        if not combinations:
            print(f"No valid combinations for {parity_name} parity.")
            return

        n_rows = 2  # Fixed for top 4 combinations (2 rows x 2 columns)
        n_cols = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
        axes = axes.flatten()  # Flatten for easier indexing

        for idx, ((J, P), count) in enumerate(combinations):
            ax = axes[idx]

            for file in files:
                # Parse the energy levels from the current file
                energy_levels = parse_CI_solutions(file)
                
                # Filter energies by the current J and P
                energies = filter_energies(energy_levels, J, P)

                # Extract the part inside {} from the file name using regex
                match = re.search(r'output_(.*)\.txt', file)
                label_part = match.group(1) if match else file  # Default to file name if no match
                
                # Determine color based on label
                color = color_map.get(label_part, "gray")  # Default color is gray
                
                # Plotting energies with a label and specific color
                ax.plot(energies, label=f'{label_part}', marker='o', color=color)
            
            # Adding labels, title, and legend
            ax.set_title(f'J={J}, P={P} (Count: {count})')
            ax.set_xlabel('State Index')
            ax.set_ylabel('Energy (a.u.)')
            ax.legend()

        # Hide unused subplots
        for idx in range(len(combinations), len(axes)):
            fig.delaxes(axes[idx])

        # Save the figure to a PDF
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(pdf_filename)
        plt.close(fig)
        print(f"{parity_name} parity plots saved to {pdf_filename}.")

    # Save even and odd parity plots to separate combined figures
    save_combined_plot(even_pdf, "Even", top_even)
    save_combined_plot(odd_pdf, "Odd", top_odd)

def plot_top_eigenvalue_differences_subplots_to_pdf(files, output_pdf="eigenvalue_differences.pdf", bins=5):
    """
    Plots histograms of the differences between the top 5 eigenvalues for all J, P combinations
    in each file, arranged in a 2x2 subplot layout, and saves them to a PDF.
    
    :param files: List of file paths to process.
    :param output_pdf: Output PDF file to save the figures.
    :param bins: Number of bins to use for the histograms.
    """
    all_differences_global = []

    # Step 1: Collect all differences globally across all files
    for file in files:
        energy_levels = parse_CI_solutions(file)
        unique_combinations = set((level['J'], level['P']) for level in energy_levels)
        
        for J, P in unique_combinations:
            energies = filter_energies(energy_levels, J, P)
            if len(energies) >= 5:
                top_5 = np.sort(energies)[-5:]
                differences = np.diff(top_5)
                all_differences_global.extend(differences)

    # Step 2: Calculate global bin edges
    bin_edges = np.linspace(min(all_differences_global), max(all_differences_global), bins + 1)

    # Step 3: Plot histograms for each file using global bin edges in a 2x2 subplot layout
    n_plots = len(files)
    n_rows, n_cols = 2, 2
    n_figs = (n_plots + n_rows * n_cols - 1) // (n_rows * n_cols)

    with PdfPages(output_pdf) as pdf:
        for fig_idx in range(n_figs):
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
            axes = axes.flatten()

            for i in range(n_rows * n_cols):
                file_idx = fig_idx * n_rows * n_cols + i
                if file_idx >= n_plots:
                    # Hide unused subplot
                    axes[i].axis('off')
                    continue

                file = files[file_idx]
                energy_levels = parse_CI_solutions(file)
                all_differences = []

                unique_combinations = set((level['J'], level['P']) for level in energy_levels)
                for J, P in unique_combinations:
                    energies = filter_energies(energy_levels, J, P)
                    if len(energies) >= 5:
                        top_5 = np.sort(energies)[-5:]
                        differences = np.diff(top_5)
                        all_differences.extend(differences)

                # Extract label from the file name using regex
                match = re.search(r'output_(.*)\.txt', file)
                label_part = match.group(1) if match else "Unknown"

                # Plot histogram for the current file using global bin edges
                axes[i].hist(all_differences, bins=bin_edges, edgecolor='k', alpha=0.7, label=label_part)
                axes[i].set_xlabel("Energy Difference (a.u.)")
                axes[i].set_ylabel("Frequency")
                axes[i].legend(loc="upper right")

            plt.tight_layout()
            pdf.savefig(fig)  # Save the current figure to the PDF
            plt.close(fig)

    print(f"Plots saved to {output_pdf}")

