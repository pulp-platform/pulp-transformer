# ----------------------------------------------------------------------
#
# File: linearProjection.py
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
import math
from collections import OrderedDict
from mako.template import Template
from mako import exceptions
from typing import Dict

def generateInputsQKV(S, E, P, H):

    bias_low = -2**15
    bias_high = 2**15 - 1

    I = torch.randint(low=-128, high=127, size=(S, E))
    W = torch.randint(low=-128, high=127, size=(P*H, E))
    B = torch.randint(low=bias_low, high=bias_high, size=(P*H,))

    return {"Input": {"data": I, "type": "int8_t"}, 
            "Weight": {"data": W, "type": "int8_t"}, 
            "Bias": {"data": B, "type": "int16_t"}}

def generateInputsO(S, E, P, H):

    bias_low = -2**15
    bias_high = 2**15 - 1

    I = torch.randint(low=-128, high=127, size=(S, P*H))
    W = torch.randint(low=-128, high=127, size=(E, P*H))
    B = torch.randint(low=bias_low, high=bias_high, size=(E,))

    return {"Input": {"data": I, "type": "int8_t"}, 
            "Weight": {"data": W, "type": "int8_t"}, 
            "Bias": {"data": B, "type": "int16_t"}}

def generateTemplateQKV(MHSAParams: Dict, requantParams: Dict, args):

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
    templateDict['l2BufferSize'] = 380000
    templateDict['l1BufferSize'] = 350000

    templateDict['S'] = S
    templateDict['E'] = E
    templateDict['P'] = P
    templateDict['H'] = H

    templateDict['inputSize'] = S*E
    templateDict['weightSize'] = P*H*E
    templateDict['biasSize'] = 2*P*H # 16b bias
    templateDict['outputSize'] = S*P*H

    templateDict['dmaTransferSize'] = 64
    templateDict['numberOfInputTransfer'] = math.ceil(templateDict['inputSize']/templateDict['dmaTransferSize'])
    templateDict['numberOfWeightTransfer'] = math.ceil(templateDict['weightSize']/templateDict['dmaTransferSize'])

    templateDict['inputVectorName'] = "testInputVectorInput"
    templateDict['weightVectorName'] = "testInputVectorWeight"
    templateDict['biasVectorName'] = "testInputVectorBias"

    templateDict['requantDiv'] = requantDiv
    templateDict['requantMul'] = requantMul

    if args.perf_cnt is None:
        templateDict['perf_counter'] = 'PI_PERF_ACTIVE_CYCLES'
    else:
        templateDict['perf_counter'] = args.perf_cnt

    l = ""
    tmpl = Template(filename=f"./TestTemplate/linearProjQKVTemplate.c")

    try:
        s = tmpl.render(verbose_log=l, **templateDict)
    except:
        print(exceptions.text_error_template().render())

    with open(f"{args.app_folder}/src/linearProjQKVTest.c", "w") as f:
        f.write(s)

def generateTemplateO(MHSAParams: Dict, requantParams: Dict, args):

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
    templateDict['l2BufferSize'] = 380000
    templateDict['l1BufferSize'] = 250000

    templateDict['S'] = S
    templateDict['E'] = E
    templateDict['P'] = P
    templateDict['H'] = H

    templateDict['inputSize'] = S*P*H
    templateDict['weightSize'] = P*H*E
    templateDict['biasSize'] = 2*E # 16b bias
    templateDict['outputSize'] = S*E

    templateDict['dmaTransferSize'] = 64
    templateDict['numberOfInputTransfer'] = math.ceil(templateDict['inputSize']/templateDict['dmaTransferSize'])
    templateDict['numberOfWeightTransfer'] = math.ceil(templateDict['weightSize']/templateDict['dmaTransferSize'])

    templateDict['inputVectorName'] = "testInputVectorInput"
    templateDict['weightVectorName'] = "testInputVectorWeight"
    templateDict['biasVectorName'] = "testInputVectorBias"

    templateDict['requantDiv'] = requantDiv
    templateDict['requantMul'] = requantMul

    if args.perf_cnt is None:
        templateDict['perf_counter'] = 'PI_PERF_ACTIVE_CYCLES'
    else:
        templateDict['perf_counter'] = args.perf_cnt

    l = ""
    tmpl = Template(filename=f"./TestTemplate/linearProjQKVTemplate.c")

    try:
        s = tmpl.render(verbose_log=l, **templateDict)
    except:
        print(exceptions.text_error_template().render())

    with open(f"{args.app_folder}/src/linearProjQKVTest.c", "w") as f:
        f.write(s)

def generateTemplateProjPULPNN(MHSAParams: Dict, requantParams: Dict, args):

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
    templateDict['l2BufferSize'] = 380000
    templateDict['l1BufferSize'] = 80000 #350000

    templateDict['S'] = S
    templateDict['E'] = E
    templateDict['P'] = P
    templateDict['H'] = H

    templateDict['inputSize'] = S*E
    templateDict['weightSize'] = P*H*E
    templateDict['biasSize'] = 2*P*H # 16b bias
    templateDict['outputSize'] = S*P*H

    templateDict['dmaTransferSize'] = 64
    templateDict['numberOfInputTransfer'] = math.ceil(templateDict['inputSize']/templateDict['dmaTransferSize'])
    templateDict['numberOfWeightTransfer'] = math.ceil(templateDict['weightSize']/templateDict['dmaTransferSize'])

    templateDict['inputVectorName'] = "testInputVectorInput"
    templateDict['weightVectorName'] = "testInputVectorWeight"
    templateDict['biasVectorName'] = "testInputVectorBias"

    templateDict['requantDiv'] = requantDiv
    templateDict['requantMul'] = requantMul

    l = ""
    tmpl = Template(filename=f"./TestTemplate/linearProjQKVTemplatePULPNN.c")

    try:
        s = tmpl.render(verbose_log=l, **templateDict)
    except:
        print(exceptions.text_error_template().render())

    with open(f"{args.app_folder}/src/linearProjQKVTestPULPNN.c", "w") as f:
        f.write(s)

def linearProjection(inputDict: Dict, requantParams: Dict, MHSAParams: Dict):

    # Unpack inputs and parameters
    I = inputDict["Input"]["data"]
    W = inputDict["Weight"]["data"]
    B = inputDict["Bias"]["data"]
    requantDiv = requantParams["div"]
    requantMul = requantParams["mul"]

    S = MHSAParams["S"]
    P = MHSAParams["P"]

    # VJ: Our kernel takes the transpose of W as input
    O = torch.matmul(I, torch.transpose(W,0,1))

    for i in range(S):
        O[i][:] += B

    O = torch.floor((O * requantMul)/(2**requantDiv))
    O = torch.clip(O, -128, 127)
    O = torch.split(O, P, dim=1)
    O = torch.stack(O)
    O = torch.squeeze(O)
    O = O.type(torch.IntTensor)

    return O

def linearProjectionQK(inputDict: Dict, requantParams: Dict, MHSAParams: Dict):
    O = linearProjection(inputDict, requantParams, MHSAParams)
    return O

def linearProjectionV(inputDict: Dict, requantParams: Dict, MHSAParams: Dict):
    O = linearProjection(inputDict, requantParams, MHSAParams)
    if len(O.shape) == 2:
        O = torch.unsqueeze(O, 0)
    return torch.transpose(O, 1, 2)

def linearProjectionO(inputDict: Dict, requantParams: Dict, MHSAParams: Dict):

    # Unpack inputs and parameters
    I = inputDict["Input"]["data"]
    W = inputDict["Weight"]["data"]
    B = inputDict["Bias"]["data"]
    requantDiv = requantParams["div"]
    requantMul = requantParams["mul"]

    S = MHSAParams["S"]
    P = MHSAParams["P"]

    # VJ: Our kernel takes the transpose of W as input
    O = torch.matmul(I, torch.transpose(W,0,1))

    for i in range(S):
        O[i][:] += B

    O = torch.floor((O * requantMul)/(2**requantDiv))
    O = torch.clip(O, -128, 127)

    return O

def linearProjectionPULPNN(inputDict: Dict, requantParams: Dict, MHSAParams: Dict):

    # Unpack inputs and parameters
    I = inputDict["Input"]["data"]
    W = inputDict["Weight"]["data"]
    B = inputDict["Bias"]["data"]
    requantDiv = requantParams["div"]
    requantMul = requantParams["mul"]

    S = MHSAParams["S"]
    P = MHSAParams["P"]

    # VJ: Our kernel takes the transpose of W as input
    O = torch.matmul(I, torch.transpose(W,0,1))

    for i in range(S):
        O[i][:] += B

    O = torch.floor((O * requantMul)/(2**requantDiv))
    O = torch.clip(O, -128, 127)
    O = O.type(torch.IntTensor)
    return O

def generateInputsOPULPNN(S, E, P, H):

    bias_low = -2**15
    bias_high = 2**15 - 1

    I = torch.randint(low=-128, high=127, size=(H, S, P))
    W = torch.randint(low=-128, high=127, size=(E, P*H))
    B = torch.randint(low=bias_low, high=bias_high, size=(E,))

    return {"Input": {"data": I, "type": "int8_t"}, 
            "Weight": {"data": W, "type": "int8_t"}, 
            "Bias": {"data": B, "type": "int16_t"}}

def linearProjectionOPULPNN(inputDict: Dict, requantParams: Dict, MHSAParams: Dict):

    # Unpack inputs and parameters
    I = inputDict["Input"]["data"]
    W = inputDict["Weight"]["data"]
    B = inputDict["Bias"]["data"]
    requantDiv = requantParams["div"]
    requantMul = requantParams["mul"]

    S = MHSAParams["S"]
    P = MHSAParams["P"]

    I = I.permute(1, 0, 2).contiguous().view(S, -1)

    # VJ: Our kernel takes the transpose of W as input
    O = torch.matmul(I, torch.transpose(W,0,1))

    for i in range(S):
        O[i][:] += B

    O = torch.floor((O * requantMul)/(2**requantDiv))
    O = torch.clip(O, -128, 127)
    O = O.type(torch.IntTensor)
    return O

def generateTemplateProjOPULPNN(MHSAParams: Dict, requantParams: Dict, args):

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
    templateDict['l2BufferSize'] = 380000
    templateDict['l1BufferSize'] = 350000

    templateDict['S'] = S
    templateDict['E'] = E
    templateDict['P'] = P
    templateDict['H'] = H

    templateDict['inputSize'] = S*P*H
    templateDict['weightSize'] = P*H*E
    templateDict['biasSize'] = 2*E # 16b bias
    templateDict['outputSize'] = S*E

    templateDict['dmaTransferSize'] = 64
    templateDict['numberOfInputTransfer'] = math.ceil(templateDict['inputSize']/templateDict['dmaTransferSize'])
    templateDict['numberOfWeightTransfer'] = math.ceil(templateDict['weightSize']/templateDict['dmaTransferSize'])

    templateDict['inputVectorName'] = "testInputVectorInput"
    templateDict['weightVectorName'] = "testInputVectorWeight"
    templateDict['biasVectorName'] = "testInputVectorBias"

    templateDict['requantDiv'] = requantDiv
    templateDict['requantMul'] = requantMul

    l = ""
    tmpl = Template(filename=f"./TestTemplate/linearProjOTemplatePULPNN.c")

    try:
        s = tmpl.render(verbose_log=l, **templateDict)
    except:
        print(exceptions.text_error_template().render())

    with open(f"{args.app_folder}/src/linearProjOTestPULPNN.c", "w") as f:
        f.write(s)



def generateTemplateSoftmax(MHSAParams: Dict, requantParams: Dict, args):

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
    templateDict['l2BufferSize'] = 70000
    templateDict['l1BufferSize'] = 70000

    templateDict['S'] = S
    templateDict['E'] = E
    templateDict['P'] = P
    templateDict['H'] = H

    templateDict['inputSize'] = S*E
    templateDict['weightSize'] = P*H*E
    templateDict['biasSize'] = 2*P*H # 16b bias
    templateDict['outputSize'] = S*P*H

    templateDict['dmaTransferSize'] = 64
    templateDict['numberOfInputTransfer'] = math.ceil(templateDict['inputSize']/templateDict['dmaTransferSize'])
    templateDict['numberOfWeightTransfer'] = math.ceil(templateDict['weightSize']/templateDict['dmaTransferSize'])

    templateDict['inputVectorName'] = "testInputVectorInput"
    templateDict['weightVectorName'] = "testInputVectorWeight"
    templateDict['biasVectorName'] = "testInputVectorBias"

    templateDict['requantDiv'] = requantDiv
    templateDict['requantMul'] = requantMul

    if args.perf_cnt is None:
        templateDict['perf_counter'] = 'PI_PERF_ACTIVE_CYCLES'
    else:
        templateDict['perf_counter'] = args.perf_cnt

    l = ""
    tmpl = Template(filename=f"./TestTemplate/iSoftmaxTemplate.c")

    try:
        s = tmpl.render(verbose_log=l, **templateDict)
    except:
        print(exceptions.text_error_template().render())

    with open(f"{args.app_folder}/src/iSoftmaxTest.c", "w") as f:  
        f.write(s)