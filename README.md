# Configuration Interaction and RMT - AMBiT

## Overview
This project provides a set of Python scripts for automating AMBiT calculations with different .matrix Hammiltonian files, and plotting analysis on random matrix theory (RMT) generated CI Hamiltonians. It automates some of the workflow, allowing for calculations on various matrix types, including standard RMT, off-diagonal RMT (od-RMT) matrices, and no-Configuratin Interaction (no-CI).

## File Descriptions
### CI_eigenvalue_analysis

This file contains functions for plotting and saving PDFs of CI eigenvalue distribution for the 4 J/P symmetries with the largest eigenvalue sets.

Key Functions:
Plot Eigenvalue Distributions
- save_combined_parity_plots(files, even_pdf="even_parity_plots.pdf", odd_pdf="odd_parity_plots.pdf")
    - files: takes list of output files
    - even_pdf: save file name
    - off_pdf: save file name

Plot Eigenvalue differences for top 5 eigenvalues
- plot_top_eigenvalue_differences_subplots_to_pdf(files, output_pdf="eigenvalue_differences.pdf", bins=5):
    - files: takes list of output files
    - output_pdf: save file name
    - bins: number of bins on histogram

- parse_CI_solutions(output):
    - internal function used to parse eigenvalue data from output files
- filter_energies(energy_levels, J, P)
    - internal function to filter energies by symmetry

### graph_ambit_output.py

This file contains functions for plotting the E1 transition data in AMBiT output .txt files. 
Generally, the only function used directly is plot_spectra.

Things to note:
- If the AMBiT input file does not ask for --print-integrals, then parse_output_o23 needs to be adjusted at lines 51 and 54 to 1 and 2. 
- The code referenced in the line above is so the second set of 'free' transition strengths are ignored.

Key Functions:
- plot_spectra(outputs, fig_name='', band=False, plot_choice='both', y='a', width=[60, 110],stems=0.1,exp=0)
    - outputs: takes list outputs of output files to plot
    - fig_name: if not empty string: figure will be saved as a pdf with fig_name as the file name
    - band: if True -  a gray band 2% around 13.5nm will be displayed on the plot
    - plot_choice: if anything except 'both', only the lorentzian will be plotted, line strengths will be omitted
    - y: choice of plotting y-axis, Line Strength 's', Einstein Coefficient ('a'), or Intensity ('i')
    - width: x axis dimensions for lorentzian generation [x_min,x_max]
    - stems: the ratio of transition weights to plot individually, e.g. 0.1 plots the top 10% only
    - exp: the scientific notation of the y-axis, defaults to automatic.

Additional Functions:
- parse_output_o23(output, y, width)
    - output: .txt AMBiT output file
    - y: Transforms transition line strength weights ('s') to Einstein Coefficients ('a'), or Intenisty ('i').
    - width: x axis dimensions for lorentzian generation [x_min,x_max].
    - Parses output file for energies and transitions.
    - Generates a lorentzian based on the weights of the transitions.
    - Returns energies, weights, total_lorentzian, x.

- process_output(idx, output, y, width)
    - Used by plot_spectra to parallelise graphing multiple outputs.
    - Runs parse_output_o23 and returns the data to plot_spectra in the neccesary format.


### random_matrices.py
This script handles the generation, manipulation, and storage of various random matrices used in CI calculations. Functions within this file allow for generating Hamiltonians of different types (RMT, NOCI, and odRMT).

Key Functions:
- create_NOCI_matrices(subfolder):
    - Reads the .matrix files in the provided subfolder (with respect to the current directory).
    - Rewrites the .matrix files with all off diagonal elements set to 0.
- create_RMT_matrices(subfolder):
    - Reads the .matrix files in the provided subfolder (with respect to the current directory).
      - Rewrites the .matrix files with all noff diagonal elements replaced with a normally distributed variable determined by the distribution of the replaced elements.
- create_odRMT_matrices(subfolder):
    - Reads the .matrix files in the provided subfolder (with respect to the current directory).
    - Rewrites the .matrix files with all non-zero off diagonal elements replaced with a normally distributed variable determined by the distribution of the replaced elements.

Additional Functions:
- Ambit_Read_Hamiltonian(filename, lower):
    -  Reads a Hamiltonian matrix from a binary file, optionally in lower-triangular form.
- off_diagonal_stats(matrix, include_zeros): 
    - Calculates statistics (mean, standard deviation) for the off-diagonal elements.
- lower_triangular_matrix(input_matrix, noci, only_nonzero): 
    - Generates a lower-triangular matrix with options for NOCI matrices and nonzero off-diagonal elements.
- collect_matrix_files(main_directory, subfolder): 
    - Gathers .matrix files in a given directory.
- create_random_matrices(matrix_files, subfolder, noci, only_nonzero): 
    - Creates and writes modified Hamiltonians with RMT, NOCI, or odRMT characteristics.

### Run_CI_NOCI_RMT_odRMT.py
- Run this script from a folder containing only the desired AMBiT input file.
- This file automates the process of running AMBiT-d four times on the same .input file with different CI Hamiltonians.
- It saves the .levels and .matrix files for each method in separate folders.

## Dependencies
- numpy
- joblib
- re
- struct
- os
- matplotlib
- shutil
- subprocess
- glob
- sys

## Getting Started

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/dirtydaz/Tin_Plasma_Thesis.git
