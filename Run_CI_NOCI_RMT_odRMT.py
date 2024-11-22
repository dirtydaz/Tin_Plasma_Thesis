import os
import shutil
import subprocess
import glob
import sys
sys.path.append('/Users/aaronfurness/AMBiT-d') ## Change to your path

from random_matrices import create_RMT_matrices, create_NOCI_matrices, create_odRMT_matrices

# from your_module import create_matrix_RMT, create_matrix_NOCI, create_matrix_odRMT

def run_ambit(input_file, output_suffix):
    """Runs the ambit program on the given input file and saves output to a specified file."""
    with open(f"output_{output_suffix}.txt", "w") as output_file:
        subprocess.run(["ambit_d", input_file], stdout=output_file, stderr=subprocess.STDOUT, check=True)

def move_files_to_folder(folder_name, file_extension):
    """Moves all files with a given extension in the current directory to a specified folder."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    for file in glob.glob(f"*{file_extension}"):
        shutil.move(file, os.path.join(folder_name, file))

def copy_files_to_folder(folder_name, file_extension):
    """Copies all files with a given extension in the current directory to a specified folder."""
    for file in glob.glob(f"*{file_extension}"):
        shutil.copy(file, os.path.join(folder_name, file))

def delete_files_with_extension(file_extension):
    """Deletes all files with a given extension in the current directory."""
    for file in glob.glob(f"*{file_extension}"):
        os.remove(file)

def find_input_file():
    """Finds the single input file with a .input extension in the current directory."""
    input_files = glob.glob("*.input")
    if len(input_files) != 1:
        raise ValueError("There should be exactly one input file with a .input extension in the directory.")
    return input_files[0]

def main():
    # Automatically find the input file
    input_file = find_input_file()

    # Step 1: Run ambit on the input file and save output to output_CI.txt
    run_ambit(input_file, "CI")

    # Step 2-3: Create CI folder, move .levels files, and copy .matrix files to CI
    move_files_to_folder("CI", ".levels")
    copy_files_to_folder("CI", ".matrix")

    # Step 4: Run create_matrix_RMT
    create_RMT_matrices('')

    # Step 5: Run ambit on the input file again and save output to output_RMT.txt
    run_ambit(input_file, "RMT")

    # Step 7: Repeat steps 2-5 with RMT folder and create_matrix_NOCI
    move_files_to_folder("RMT", ".levels")
    copy_files_to_folder("RMT", ".matrix")
    create_NOCI_matrices('')
    run_ambit(input_file, "NOCI")

    # Step 8: Repeat steps 2-3 with NOCI folder
    move_files_to_folder("NOCI", ".levels")
    copy_files_to_folder("NOCI", ".matrix")

    # Step 9: Delete matrix files in the main folder
    delete_files_with_extension(".matrix")

    # Step 10: Copy matrix files from CI back to the main folder
    for file in glob.glob("CI/*.matrix"):
        shutil.copy(file, ".")

    # Step 11: Run create_matrix_odRMT
    create_odRMT_matrices('')

    # Step 12: Run ambit on the input file one final time and save output to output_odRMT.txt
    run_ambit(input_file, "odRMT")

    # Step 13: Repeat steps 2-3 with odRMT folder
    move_files_to_folder("odRMT", ".levels")
    copy_files_to_folder("odRMT", ".matrix")

if __name__ == "__main__":
    main()
