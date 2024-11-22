import numpy as np
import struct
import os
import re

def Ambit_Read_Hamiltonian(filename, lower=False):
    # Open the file in binary read mode
    with open(filename, 'rb') as file:
        # Read the first integer (dimension of the matrix)
        dim = struct.unpack('i', file.read(4))[0]
        
        # Initialize an empty list to hold the rows of the lower triangle
        hamiltonian = []
        
        # Read the lower triangle of the matrix
        for i in range(1, dim + 1):
            row = struct.unpack('d' * i, file.read(8 * i))
            hamiltonian.append(list(row))
        
        # Convert the list of rows to a numpy array and pad to make it a square matrix
        hamiltonian = np.array([row + [0] * (dim - len(row)) for row in hamiltonian])

        if lower == False:
            # Create the symmetric matrix
            hamiltonian = hamiltonian + hamiltonian.T - np.diag(np.diag(hamiltonian))
        else:
            return hamiltonian
        
        return hamiltonian

def off_diagonal_stats(matrix, include_zeros=False):
    n = matrix.shape[0]
    diags = np.diag(matrix)
    
    # Create a mask for off-diagonal elements
    mask = ~np.eye(n, dtype=bool)
    
    # Get off-diagonal elements
    off_diagonal = matrix[mask]
    
    if include_zeros:
        data = off_diagonal
    else:
        data = off_diagonal[off_diagonal != 0]
    
    if data.size == 0:
        mean = 0.0
        std = 0.0
    else:
        mean = np.mean(data)
        std = np.std(data)
    
    return n, diags, mean, std

def Ambit_Write_Hamiltonian(filename, matrix):
    # Ensure the matrix is square and lower triangular
    dim = matrix.shape[0]
    if matrix.shape != (dim, dim):
        raise ValueError("Matrix must be square")
    
    # Open the file in binary write mode
    with open(filename, 'wb') as file:
        # Write the dimension of the matrix as an integer
        file.write(struct.pack('i', dim))
        
        # Write the lower triangle of the matrix
        for i in range(dim):
            # Get the lower triangular part of this row
            row = matrix[i, :i+1]
            # Write the row as doubles
            file.write(struct.pack('d' * len(row), *row))
                
    return

def lower_triangular_matrix(input_matrix, noci=False, only_nonzero=False):
    n = input_matrix.shape[0]
    if n <= 3:
        diags = np.diag(input_matrix)
        matrix = np.zeros((n, n))
        np.fill_diagonal(matrix, diags)
        return matrix

    diags = np.diag(input_matrix)
    matrix = np.zeros((n, n))
    np.fill_diagonal(matrix, diags)

    if noci:
        return matrix

    if only_nonzero:
        n, diags_unused, mean, std = off_diagonal_stats(input_matrix, include_zeros=False)
        if std == 0.0:
            # If std is zero, replace with mean
            for i in range(1, n):
                for j in range(i):
                    if input_matrix[i, j] != 0:
                        matrix[i, j] = mean
                    else:
                        matrix[i, j] = 0.0
        else:
            # Replace only the non-zero off-diagonal elements
            for i in range(1, n):
                for j in range(i):
                    if input_matrix[i, j] != 0:
                        matrix[i, j] = np.random.normal(mean, std)
                    else:
                        matrix[i, j] = 0.0
    else:
        n, diags_unused, mean, std = off_diagonal_stats(input_matrix, include_zeros=True)
        if std == 0.0:
            # If std is zero, fill with mean
            for i in range(1, n):
                matrix[i, :i] = mean
        else:
            # Replace all off-diagonal elements
            for i in range(1, n):
                matrix[i, :i] = np.random.normal(mean, std, i)
    return matrix

def collect_matrix_files(main_directory, subfolder):
    # Construct the full path to the subfolder
    subfolder_path = os.path.join(main_directory, subfolder)
    
    # Check if the subfolder exists
    if not os.path.exists(subfolder_path):
        print(f"Error: The subfolder '{subfolder}' does not exist in '{main_directory}'")
        return []

    # Initialize an empty list to store the file names
    matrix_files = []

    # Compile the regular expression pattern
    pattern = re.compile(r'\d+[eo]\.matrix$')

    # Iterate through files in the subfolder
    for filename in os.listdir(subfolder_path):
        # Check if the file matches the pattern
        if pattern.search(filename):
            # Add the file name to the list
            matrix_files.append(filename)

    return matrix_files

def create_random_matrices(matrix_files, subfolder, noci=False, only_nonzero=False):
    og_matrices = []
    rmt_matrices = []
    current_directory = os.getcwd()
    for file in matrix_files:
        file_path = os.path.join(current_directory, subfolder, file)
        # Read Hamiltonian Matrix
        H = Ambit_Read_Hamiltonian(file_path)

        # Create new H-Matrix
        H_RMT = lower_triangular_matrix(H, noci=noci, only_nonzero=only_nonzero)
            
        # Write new H-Matrix to same .matrix file
        Ambit_Write_Hamiltonian(file_path, H_RMT) 

        # Save original and new matrices
        og_matrices.append(H)
        rmt_matrices.append(H_RMT)
    return og_matrices, rmt_matrices


def create_odRMT_matrices(subfolder):
    current_directory = os.getcwd()
    matrix_folder = subfolder
    ## Run function to collect matrix files of relevant folder
    matrix_files = collect_matrix_files(current_directory, matrix_folder)
    ## Run function to create and write matrices with off diagonals RMT
    matrix_OG, matrix_odRMT = create_random_matrices(matrix_files, matrix_folder, only_nonzero=True)

def create_NOCI_matrices(subfolder):
    current_directory = os.getcwd()
    matrix_folder = subfolder
    ## Run function to collect matrix files of relevant folder
    matrix_files = collect_matrix_files(current_directory, matrix_folder)
    ## Run function to create and write matrices with off diagonals = 0
    matrix_OG, matrix_RMT = create_random_matrices(matrix_files, matrix_folder, True)

def create_RMT_matrices(subfolder):
    current_directory = os.getcwd()
    matrix_folder = subfolder
    ## Run function to collect matrix files of relevant folder
    matrix_files = collect_matrix_files(current_directory, matrix_folder)
    ## Run function to create and write matrices with off diagonals RMT
    matrix_OG, matrix_RMT = create_random_matrices(matrix_files, matrix_folder)