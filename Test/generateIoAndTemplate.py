# ----------------------------------------------------------------------
#
# File: generateGoldenOutput.py
#
# Last edited: 29.08.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author:
# - Victor Jung, jungvi@iis.ee.ethz.ch, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import argparse
from typing import Dict, List
import shutil
import os
import yaml

import GoldenModel


def generateIoAndTemplate(args):

    S = args.MHSA_params[0]
    E = args.MHSA_params[1]
    P = args.MHSA_params[2]
    H = args.MHSA_params[3]
    requantDiv = 16
    requantMul = 385

    # Pack parameters
    MHSAParams = {"S": S, "E": E, "P": P, "H": H}
    requantParams = {"div": requantDiv, "mul": requantMul}

    with open('./testConfig.yml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    testName = config["testToRun"][args.test_idx]

    if config[testName]["inputGen"] != "None":
        inputGen = getattr(GoldenModel, config[testName]["inputGen"])
        inputDict = inputGen(S, E, P, H)
        generateHeaders(inputDict, args)

    if config[testName]["goldenKernel"] != "None":
        goldenKernel = getattr(GoldenModel, config[testName]["goldenKernel"])
        output = goldenKernel(inputDict, requantParams, MHSAParams)
        torch.save(output, f'{args.app_folder}/testGoldenOutput.pt')

    torch.manual_seed(config["seed"])

    headerToCopy = ["dory.h", "mchan_test.h", "pulp_nn_kernels.h", "pulp_nn_utils.h", "thorir_dma.h"]
    srcToCopy = ["dory.c", "iSoftmax.c", "thorir_dma.c"]

    if args.kernel_name != "MHSA":
        srcToCopy += ["linearQK_4x2_H.c", "linearV_4x2_H.c", "matmulSoftmax_4x2_S.c", 
                      "matmul_4x2_S.c", "linearO_4x2_H.c", "matmulSoftmax_FWA_v3_H.c", 
                      "matmulSoftmax_FWA_v3_S.c", "pulp_nn_linear_i8_i8_i8.c", "matmulSoftmax_4x2_H.c", 
                      "matmul_4x2_H.c"]
    else:
        srcToCopy += [args.kernel_name + ".c"]

    copyFilesToApp(headerToCopy, srcToCopy, args.app_folder)

    templateGen = getattr(GoldenModel, config[testName]["templateGen"])
    templateGen(MHSAParams, requantParams, args)

def generateHeaders(tensorDict: Dict, args):

    retStr = ""
    for name, tensorDict in tensorDict.items():
        tensor = tensorDict["data"]
        tensor = tensor.flatten()
        tensor = tensor.numpy()
        tensor = tensor.astype(int)

        retStr += '#include "pmsis.h"\n\n'
        retStr += f"{tensorDict['type']} testInputVector{name}[] ="
        retStr += "{"
        list_str = (", ").join([str(x) for x in tensor])
        retStr += list_str
        retStr += "};\n\n"

    f = open(f'{args.app_folder}/inc/testInput.h', "w")
    f.write(retStr)
    f.close()

def copyFilesToApp(headerToCopy: List[str], srcToCopy: List[str], app_folder):

    # Ensure the destination directories exist
    os.makedirs(os.path.join(app_folder, "inc"), exist_ok=True)
    os.makedirs(os.path.join(app_folder, "src"), exist_ok=True)

    # Copy header files
    for header in headerToCopy:
        source_path_kernel = os.path.join("../Kernel/includes", header) 
        source_path_helpers = os.path.join("./Helpers", header)

        # Determine which source path exists
        if os.path.exists(source_path_helpers):
            source_path = source_path_helpers
        elif os.path.exists(source_path_kernel):
            source_path = source_path_kernel
        else:
            print(f"Warning: Source file {header} not found in Helpers or ../Kernel/src.")
            continue

        dest_path = os.path.join(app_folder, "inc", header)
        shutil.copy2(source_path, dest_path)

    # Copy source files
    for src in srcToCopy:
        source_path_helpers = os.path.join("./Helpers", src)   # Look in Helpers
        source_path_kernel = os.path.join("../Kernel/src", src)  # Look in Kernel/src

        # Determine which source path exists
        if os.path.exists(source_path_helpers):
            source_path = source_path_helpers
        elif os.path.exists(source_path_kernel):
            source_path = source_path_kernel
        else:
            print(f"Warning: Source file {src} not found in Helpers or ../Kernel/src.")
            continue

        dest_path = os.path.join(app_folder, "src", src)
        shutil.copy2(source_path, dest_path)
    
    # Copy Makefile
    source_path = os.path.join("./Helpers", "Makefile")
    dest_path = os.path.join(app_folder, "Makefile")
    shutil.copy2(source_path, dest_path)

def generateIOHeadersARM(args):

    S = args.MHSA_params[0]
    E = args.MHSA_params[1]
    P = args.MHSA_params[2]
    H = args.MHSA_params[3]
    requantDiv = 16
    requantMul = 385

    # Pack parameters
    MHSAParams = {"S": S, "E": E, "P": P, "H": H}
    requantParams = {"div": requantDiv, "mul": requantMul}

    with open('./testConfig.yml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    testName = config["testToRun"][args.test_idx]

    inputGen = getattr(GoldenModel, config[testName]["inputGen"])
    templateGen = getattr(GoldenModel, config[testName]["templateGen"])
    goldenKernel = getattr(GoldenModel, config[testName]["goldenKernel"])

    torch.manual_seed(config["seed"])

    inputDict = inputGen(S, E, P, H)
    output = goldenKernel(inputDict, requantParams, MHSAParams)

    # Generate headers ARM

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process MHSA parameters and other options.")

    # MHSA_params expects exactly 4 arguments
    parser.add_argument('--MHSA_params', type=int,  nargs=4, metavar=('S', 'E', 'P', 'H'), required=True, help='Provide exactly 4 arguments for MHSA parameters.')
    # kernel_name expects a single argument
    parser.add_argument('--kernel_name', type=str, required=True, help='Kernel name.')
    # app_folder expects a single argument
    parser.add_argument('--app_folder', type=str, required=True, help='Application folder.')
    parser.add_argument('--board', type=str, required=True, help='Board to use.')
    parser.add_argument('--test_idx', type=int, required=True, help='Index of the test to run.')
    parser.add_argument('--ARM', type=bool, help='Run on ARM.')
    parser.add_argument('--perf_cnt', type=str)

    args = parser.parse_args()

    if args.ARM:
        generateIOHeadersARM(args)
    else:
        generateIoAndTemplate(args)
