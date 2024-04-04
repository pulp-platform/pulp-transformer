# ----------------------------------------------------------------------
#
# File: extractProfilingData.py
#
# Last edited: 28.09.2023
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
 

import argparse

def extract_profiling_data(log_file, result_file, args):

    S = args.MHSA_params[0]
    E = args.MHSA_params[1]
    P = args.MHSA_params[2]
    H = args.MHSA_params[3]

    log_perf_counter = True

    test_name_SEPH = ['projQK', 'projV', 'projO', 'projPULPNN', 'projOPULPNN']

    MACs = 0
    if args.test_name in test_name_SEPH:
        MACs = S*E*P*H
    else:
        MACs = H*S*S*P

    with open(log_file, 'r') as f_log:
        with open(result_file, 'a') as f_result:
            sequential_cycles = 0
            execution_cycles = 0
            perf_counter = ''
            for line in f_log:
                if "Kernel Execution:" in line:
                    execution_cycles += int(line.split("Kernel Execution:")[1].strip())
                if "Sequential:" in line:
                    sequential_cycles += int(line.split("Sequential:")[1].strip())
                if "PI_PERF" in line:
                    perf_counter = line
            if execution_cycles == 0:
                execution_cycles = 1
            log_str = f"{args.test_name}:(S={S},E={E},P={P},H={H}):{execution_cycles}:{(MACs/execution_cycles):.2f}:{sequential_cycles}\n"
            
            if log_perf_counter:
                log_str = log_str.strip() + ":" + perf_counter.strip() + '\n'
            f_result.write(log_str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract profiling data from log.')
    parser.add_argument('--log_file', type=str, required=True, help='Path to the log file.')
    parser.add_argument('--MHSA_params', nargs=4, type=int, help='MHSA parameters (S E P H).')
    parser.add_argument('--kernel_name', type=str, required=True, help='Name of the kernel.')
    parser.add_argument('--test_name', type=str, required=True, help='Name of the test.')
    parser.add_argument('--result_file', type=str, required=True, help='Path to the result file.')

    args = parser.parse_args()
    extract_profiling_data(args.log_file, args.result_file, args)
