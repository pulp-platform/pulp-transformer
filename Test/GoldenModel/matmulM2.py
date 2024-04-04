# ----------------------------------------------------------------------
#
# File: matmulM2.py
#
# Last edited: 14.09.2023
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
import math
from .iSoftmax import ibertSoftmax
from typing import Dict
from collections import OrderedDict
from mako.template import Template
from mako import exceptions


def generateInputsM2(S, E, P, H):

    bias_low = -2**15
    bias_high = 2**15 - 1

    A = torch.randint(low=-128, high=127, size=(H*S, S))
    B = torch.randint(low=-128, high=127, size=(H, P, S))

    return {"A": {"data": A, "type": "int8_t"}, 
            "B": {"data": B, "type": "int8_t"}}

def generateTemplateM2(MHSAParams: Dict, requantParams: Dict, args):

    # Unpack params
    S = MHSAParams["S"]
    E = MHSAParams["E"]
    P = MHSAParams["P"]
    H = MHSAParams["H"]
    requantDiv = requantParams["div"]
    requantMul = requantParams["mul"]

    templateDict = OrderedDict()

    templateDict["kernelName"] = args.kernel_name
    templateDict["testInputHeaderName"] = "testInput"

    templateDict['fcFrequency'] = 100000000
    templateDict['clFrequency'] = 100000000
    templateDict['l2BufferSize'] = 700000
    templateDict['l1BufferSize'] = 400000

    templateDict['S'] = S
    templateDict['E'] = E
    templateDict['P'] = P
    templateDict['H'] = H

    templateDict['sizeA'] = H*S*S
    templateDict['sizeB'] = H*S*P
    templateDict['outputSize'] = H*S*P

    templateDict['dmaTransferSize'] = 64
    templateDict['numberOfTransferA'] = math.ceil(templateDict['sizeA']/templateDict['dmaTransferSize'])
    templateDict['numberOfTransferB'] = math.ceil(templateDict['sizeB']/templateDict['dmaTransferSize'])

    templateDict['vectorNameA'] = "testInputVectorA"
    templateDict['vectorNameB'] = "testInputVectorB"

    templateDict['requantDiv'] = requantDiv
    templateDict['requantMul'] = requantMul

    if args.perf_cnt is None:
        templateDict['perf_counter'] = 'PI_PERF_ACTIVE_CYCLES'
    else:
        templateDict['perf_counter'] = args.perf_cnt

    l = ""
    tmpl = Template(filename=f"./TestTemplate/matmulM2Template.c")

    try:
        s = tmpl.render(verbose_log=l, **templateDict)
    except:
        print(exceptions.text_error_template().render())

    with open(f"{args.app_folder}/src/matmulM2.c", "w") as f:
        f.write(s)

def matmulM2(inputDict: Dict, requantParams: Dict, MHSAParams: Dict):

    # Unpack inputs and parameters
    A = inputDict["A"]["data"]
    V = inputDict["B"]["data"]
    requant_div = requantParams["div"]
    requant_mul = requantParams["mul"]
    S = MHSAParams["S"]
    P = MHSAParams["P"]
    H = MHSAParams["H"]

    V = V.transpose(1, 2)

    # Special layout for the input: head interleaved fashioned
    A2 = torch.zeros((H, S, S))
    for s in range(S):
        for h in range(H):
            A2[h, s, :] = A[s*H + h, :]
    A = A2.to(torch.int64)

    # Convert A to uint8
    mask = A < 0
    A[mask] += 256

    output = []
    for i in range(H):
        output.append(torch.matmul(A[i, :, :], V[i, :, :]))
        output[i] = torch.floor((output[i] * requant_mul)/(2**requant_div))
        output[i] = torch.clip(output[i], -128, 127)

    A = torch.stack(output)

    # Special layout for the output: head interleaved fashioned
    out = torch.zeros((H*S, P), dtype=torch.int8)
    for s in range(S):
        for h in range(H):
            out[s*H + h, :] = A[h, s, :]

    return out


def generateInputsM2PULPNN(S, E, P, H):

    bias_low = -2**15
    bias_high = 2**15 - 1

    A = torch.randint(low=-128, high=127, size=(H, S, S))
    B = torch.randint(low=-128, high=127, size=(S, P*H))

    return {"A": {"data": A, "type": "int8_t"}, 
            "B": {"data": B, "type": "int8_t"}}

def generateTemplateM2PULPNN(MHSAParams: Dict, requantParams: Dict, args):

    # Unpack params
    S = MHSAParams["S"]
    E = MHSAParams["E"]
    P = MHSAParams["P"]
    H = MHSAParams["H"]
    requantDiv = requantParams["div"]
    requantMul = requantParams["mul"]

    templateDict = OrderedDict()

    templateDict["kernelName"] = args.kernel_name
    templateDict["testInputHeaderName"] = "testInput"

    templateDict['fcFrequency'] = 100000000
    templateDict['clFrequency'] = 100000000
    templateDict['l2BufferSize'] = 700000
    templateDict['l1BufferSize'] = 400000

    templateDict['S'] = S
    templateDict['E'] = E
    templateDict['P'] = P
    templateDict['H'] = H

    templateDict['sizeA'] = H*S*S
    templateDict['sizeB'] = H*S*P
    templateDict['outputSize'] = H*S*P

    templateDict['dmaTransferSize'] = 64
    templateDict['numberOfTransferA'] = math.ceil(templateDict['sizeA']/templateDict['dmaTransferSize'])
    templateDict['numberOfTransferB'] = math.ceil(templateDict['sizeB']/templateDict['dmaTransferSize'])

    templateDict['vectorNameA'] = "testInputVectorA"
    templateDict['vectorNameB'] = "testInputVectorB"

    templateDict['requantDiv'] = requantDiv
    templateDict['requantMul'] = requantMul

    if args.perf_cnt is None:
        templateDict['perf_counter'] = 'PI_PERF_ACTIVE_CYCLES'
    else:
        templateDict['perf_counter'] = args.perf_cnt

    l = ""
    tmpl = Template(filename=f"./TestTemplate/matmulM2TemplatePULPNN.c")

    try:
        s = tmpl.render(verbose_log=l, **templateDict)
    except:
        print(exceptions.text_error_template().render())

    with open(f"{args.app_folder}/src/matmulM2.c", "w") as f:
        f.write(s)

def matmulM2PULPNN(inputDict: Dict, requantParams: Dict, MHSAParams: Dict):

    # Unpack inputs and parameters
    A = inputDict["A"]["data"]
    V = inputDict["B"]["data"]
    requant_div = requantParams["div"]
    requant_mul = requantParams["mul"]
    S = MHSAParams["S"]
    P = MHSAParams["P"]
    H = MHSAParams["H"]

    V = torch.split(V, P, dim=1)
    V = torch.stack(V)
    V = torch.squeeze(V)

    # Convert A to uint8
    # mask = A < 0
    # A[mask] += 256

    output = []
    for i in range(H):
        output.append(torch.matmul(A[i, :, :], V[i, :, :]))
        output[i] = torch.floor((output[i] * requant_mul)/(2**requant_div))
        output[i] = torch.clip(output[i], -128, 127)

    A = torch.stack(output)
    
    return A