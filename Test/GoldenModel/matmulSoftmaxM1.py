# ----------------------------------------------------------------------
#
# File: gemmSoftmax.py
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


def generateInputsM1(S, E, P, H):

    bias_low = -2**15
    bias_high = 2**15 - 1

    A = torch.randint(low=-128, high=127, size=(H, S, P))
    B = torch.randint(low=-128, high=127, size=(H, S, P))

    return {"A": {"data": A, "type": "int8_t"}, 
            "B": {"data": B, "type": "int8_t"}}

def generateTemplateM1(MHSAParams: Dict, requantParams: Dict, args):

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
    templateDict['l1BufferSize'] = 500000

    templateDict['S'] = S
    templateDict['E'] = E
    templateDict['P'] = P
    templateDict['H'] = H

    templateDict['sizeA'] = H*S*P
    templateDict['sizeB'] = H*S*P
    templateDict['outputSize'] = H*S*S

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
    tmpl = Template(filename=f"./TestTemplate/matmulSoftmaxM1Template.c")

    try:
        s = tmpl.render(verbose_log=l, **templateDict)
    except:
        print(exceptions.text_error_template().render())

    with open(f"{args.app_folder}/src/matmulSoftmaxM1.c", "w") as f:
        f.write(s)

def matmulSoftmaxM1(inputDict: Dict, requantParams: Dict, MHSAParams: Dict):

    # Unpack inputs and parameters
    Q = inputDict["A"]["data"]
    K = inputDict["B"]["data"]
    pre_softmax_requant_div = requantParams["div"]
    pre_softmax_requant_mul = requantParams["mul"]
    S = MHSAParams["S"]
    H = MHSAParams["H"]

    K = torch.transpose(K,1,2)
    
    output_head_list = []
    output = []
    for i in range(H):
        output_head_list.append(torch.matmul(Q[i, :, :], K[i, :, :]))
        output_head_list[i] = torch.floor((output_head_list[i] * pre_softmax_requant_mul)/(2**pre_softmax_requant_div))
        output_head_list[i] = torch.clip(output_head_list[i], -128, 127)
        output.append(ibertSoftmax(output_head_list[i]))

    A = torch.stack(output)

    # Special layout for the output: head interleaved fashioned
    out = torch.zeros((H*S, S), dtype=torch.int8)
    for s in range(S):
        for h in range(H):
            out[s*H + h, :] = A[h, s, :]

    return out.to(torch.uint8)


def generateInputsM1PULPNN(S, E, P, H):

    bias_low = -2**15
    bias_high = 2**15 - 1

    A = torch.randint(low=-128, high=127, size=(S, P*H))
    B = torch.randint(low=-128, high=127, size=(S, P*H))

    return {"A": {"data": A, "type": "int8_t"}, 
            "B": {"data": B, "type": "int8_t"}}

def generateTemplateM1PULPNN(MHSAParams: Dict, requantParams: Dict, args):

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
    templateDict['l1BufferSize'] = 500000

    templateDict['S'] = S
    templateDict['E'] = E
    templateDict['P'] = P
    templateDict['H'] = H

    templateDict['sizeA'] = H*S*P
    templateDict['sizeB'] = H*S*P
    templateDict['outputSize'] = H*S*S

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
    tmpl = Template(filename=f"./TestTemplate/matmulSoftmaxM1TemplatePULPNN.c")

    try:
        s = tmpl.render(verbose_log=l, **templateDict)
    except:
        print(exceptions.text_error_template().render())

    with open(f"{args.app_folder}/src/matmulSoftmaxM1.c", "w") as f:
        f.write(s)

def matmulSoftmaxM1PULPNN(inputDict: Dict, requantParams: Dict, MHSAParams: Dict):

    # Unpack inputs and parameters
    Q = inputDict["A"]["data"]
    K = inputDict["B"]["data"]
    pre_softmax_requant_div = requantParams["div"]
    pre_softmax_requant_mul = requantParams["mul"]
    S = MHSAParams["S"]
    P = MHSAParams["P"]
    H = MHSAParams["H"]

    Q = torch.split(Q, P, dim=1)
    Q = torch.stack(Q)
    Q = torch.squeeze(Q)

    K = torch.split(K, P, dim=1)
    K = torch.stack(K)
    K = torch.squeeze(K)

    K = torch.transpose(K,1,2)

    output_head_list = []
    output = []
    for i in range(H):
        output_head_list.append(torch.matmul(Q[i, :, :], K[i, :, :]))
        output_head_list[i] = torch.floor((output_head_list[i] * pre_softmax_requant_mul)/(2**pre_softmax_requant_div))
        output_head_list[i] = torch.clip(output_head_list[i], -128, 127)
        output.append(ibertSoftmax(output_head_list[i]))

    A = torch.stack(output)

    return A.to(torch.uint8)