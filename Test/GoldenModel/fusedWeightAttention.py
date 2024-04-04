# ----------------------------------------------------------------------
#
# File: fusedWeightAttention.py
#
# Last edited: 23.10.2023
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


def generateInputsFWA(S, E, P, H):

    bias_low = -2**15
    bias_high = 2**15 - 1

    I = torch.randint(low=-128, high=127, size=(S, E))
    W = torch.randint(low=-128, high=127, size=(H, E, E))
    B = torch.randint(low=bias_low, high=bias_high, size=(H,E))

    return {"I": {"data": I, "type": "int8_t"}, 
            "W": {"data": W, "type": "int8_t"},
           "Bias": {"data": B, "type": "int16_t"}}

def generateTemplateFWA(MHSAParams: Dict, requantParams: Dict, args):

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

    templateDict['fcFrequency'] = 370*1000*1000
    templateDict['clFrequency'] = 370*1000*1000
    templateDict['l2BufferSize'] = 700000
    templateDict['l1BufferSize'] = 80000

    templateDict['S'] = S
    templateDict['E'] = E
    templateDict['P'] = P
    templateDict['H'] = H

    templateDict['sizeI'] = S*E
    templateDict['biasSize'] = 2*H*E
    templateDict['sizeW'] = H*E*E
    templateDict['outputSize'] = H*S*S

    templateDict['dmaTransferSize'] = 64
    templateDict['numberOfTransferI'] = math.ceil(templateDict['sizeI']/templateDict['dmaTransferSize'])
    templateDict['numberOfTransferW'] = math.ceil(templateDict['sizeW']/templateDict['dmaTransferSize'])

    templateDict['vectorNameI'] = "testInputVectorI"
    templateDict['biasVectorName'] = "testInputVectorBias"
    templateDict['vectorNameW'] = "testInputVectorW"

    templateDict['requantDiv'] = requantDiv
    templateDict['requantMul'] = requantMul

    l = ""
    tmpl = Template(filename=f"./TestTemplate/matmulSoftmaxFWATemplate.c")

    try:
        s = tmpl.render(verbose_log=l, **templateDict)
    except:
        print(exceptions.text_error_template().render())

    with open(f"{args.app_folder}/src/matmulSoftmaxFWA.c", "w") as f:
        f.write(s)

def matmulSoftmaxFWA(inputDict: Dict, requantParams: Dict, MHSAParams: Dict):

    # Unpack inputs and parameters
    I = inputDict["I"]["data"]
    W = inputDict["W"]["data"]
    B = inputDict["Bias"]["data"]
    pre_proj_requant_div = requantParams["div"]
    pre_proj_requant_mul = requantParams["mul"]
    post_proj_requant_div = requantParams["div"]
    post_proj_requant_mul = requantParams["mul"]
    S = MHSAParams["S"]
    H = MHSAParams["H"]

    W = W.transpose(1, 2)

    output = []
    for i in range(H):
        I_star = torch.matmul(I, W[i, :, :])
        for s in range(S):
            I_star[s][:] += B[i][:]

        I_star = torch.floor((I_star * pre_proj_requant_mul)/(2**pre_proj_requant_div))
        I_star = torch.clip(I_star, -128, 127).type(torch.LongTensor)
        A = torch.matmul( I_star, I.transpose(0, 1))
        A = torch.floor((A * post_proj_requant_mul)/(2**post_proj_requant_div))
        A = torch.clip(A, -128, 127)
        output.append(ibertSoftmax(A))

    A = torch.stack(output)

    # Special layout for the output: head interleaved fashioned
    # out = torch.zeros((H*S, S), dtype=torch.int8)
    # for s in range(S):
    #     for h in range(H):
    #         out[s*H + h, :] = A[h, s, :]

    out = A
    return out.to(torch.uint8)