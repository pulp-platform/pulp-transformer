import torch
import numpy as np
import argparse
import sys

def extract_array_from_log(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if "Output:" in line:
            # Extract the next line and split it by ', '
            values = lines[i+1].strip().split(', ')
            values[-1] = values[-1][:-1]
            return torch.from_numpy(np.array(values, dtype=int))
    return None



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Compare outputs based on provided parameters.")

    # log_file expects a single argument
    parser.add_argument('--log_file', type=str, required=True, help='Path to the log file.')

    # MHSA_params expects exactly 4 arguments
    parser.add_argument('--MHSA_params', type=int, nargs=4, metavar=('S', 'E', 'P', 'H'), required=True, help='Provide exactly 4 arguments for MHSA parameters.')
    # kernel_name expects a single argument
    parser.add_argument('--kernel_name', type=str, required=True, help='Kernel name.')
    # app_folder expects a single argument
    parser.add_argument('--app_folder', type=str, required=True, help='Application folder.')

    args = parser.parse_args()

    S = args.MHSA_params[0]
    E = args.MHSA_params[1]
    P = args.MHSA_params[2]
    H = args.MHSA_params[3]

    array = extract_array_from_log(args.log_file)

    if array is not None:
        print("Output collected successfully.")
    else:
        print("Array not found in the given log file.")
        # raise ValueError
        sys.exit()

    # Load the golden output
    golden_output = torch.load(f'{args.app_folder}/testGoldenOutput.pt').flatten()

    print("First elements of the output:")
    print("\tGolden output: \t", golden_output[:32].tolist())
    print("\tOutput: \t", array[:32].tolist())
    # print("")

    sucess = torch.eq(golden_output, array).all()
    
    if sucess:
        print("\U00002705 Test passed!\n")
    else:
        print("\U0000274C Test failed.\n")

        isEqualMask = torch.eq(golden_output, array)
        nonEqualIndices = torch.nonzero(~isEqualMask)

        print(isEqualMask)
        # print(nonEqualIndices)
        
        if nonEqualIndices.shape[0] > 10:
            print("Showing only the first 10 non-equal values.")
            nonEqualIndices = nonEqualIndices[:10]
        
        for indices in nonEqualIndices:
            print(f"{indices.item()} Golden / Output: {golden_output[indices].item()} / {array[indices].item()}")
            print("")
        
        raise ValueError