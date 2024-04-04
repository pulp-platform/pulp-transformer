# ----------------------------------------------------------------------
#
# File: MHSA.py
#
# Last edited: 06.11.2023
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
 
import math
from typing import Dict
from collections import OrderedDict
from mako.template import Template
from mako import exceptions


def generateTemplateMHSA(MHSAParams: Dict, requantParams: Dict, args):

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

    templateDict['sizeA'] = H*S*P
    templateDict['sizeB'] = H*S*P
    templateDict['sizeBias'] = 2*P*H # 16b bias
    templateDict['outputSize'] = H*S*S

    templateDict['dmaTransferSize'] = 64
    templateDict['numberOfTransferA'] = math.ceil(templateDict['sizeA']/templateDict['dmaTransferSize'])
    templateDict['numberOfTransferB'] = math.ceil(templateDict['sizeB']/templateDict['dmaTransferSize'])

    templateDict['vectorNameA'] = "testInputVectorA"
    templateDict['vectorNameB'] = "testInputVectorB"

    templateDict['requantDiv'] = requantDiv
    templateDict['requantMul'] = requantMul

    l = ""
    tmpl = Template(filename=f"./TestTemplate/MHSATemplate.c")

    try:
        s = tmpl.render(verbose_log=l, **templateDict)
    except:
        print(exceptions.text_error_template().render())

    with open(f"{args.app_folder}/src/MHSA.c", "w") as f:
        f.write(s)

def generateTemplateMHSAFWA(MHSAParams: Dict, requantParams: Dict, args):

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

    templateDict['sizeA'] = H*S*P
    templateDict['sizeB'] = H*S*P
    templateDict['sizeBias'] = 2*P*H # 16b bias
    templateDict['outputSize'] = H*S*S

    templateDict['dmaTransferSize'] = 64
    templateDict['numberOfTransferA'] = math.ceil(templateDict['sizeA']/templateDict['dmaTransferSize'])
    templateDict['numberOfTransferB'] = math.ceil(templateDict['sizeB']/templateDict['dmaTransferSize'])

    templateDict['vectorNameA'] = "testInputVectorA"
    templateDict['vectorNameB'] = "testInputVectorB"

    templateDict['requantDiv'] = requantDiv
    templateDict['requantMul'] = requantMul

    l = ""
    tmpl = Template(filename=f"./TestTemplate/MHSAFWATemplate.c")

    try:
        s = tmpl.render(verbose_log=l, **templateDict)
    except:
        print(exceptions.text_error_template().render())

    with open(f"{args.app_folder}/src/MHSAFWA.c", "w") as f:
        f.write(s)


def generateTemplateMHSAPULPNN(MHSAParams: Dict, requantParams: Dict, args):

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

    templateDict['sizeA'] = H*S*P
    templateDict['sizeB'] = H*S*P
    templateDict['sizeBias'] = 2*P*H # 16b bias
    templateDict['outputSize'] = H*S*S

    templateDict['dmaTransferSize'] = 64
    templateDict['numberOfTransferA'] = math.ceil(templateDict['sizeA']/templateDict['dmaTransferSize'])
    templateDict['numberOfTransferB'] = math.ceil(templateDict['sizeB']/templateDict['dmaTransferSize'])

    templateDict['vectorNameA'] = "testInputVectorA"
    templateDict['vectorNameB'] = "testInputVectorB"

    templateDict['requantDiv'] = requantDiv
    templateDict['requantMul'] = requantMul

    l = ""
    tmpl = Template(filename=f"./TestTemplate/MHSAPULPNNTemplate.c")

    try:
        s = tmpl.render(verbose_log=l, **templateDict)
    except:
        print(exceptions.text_error_template().render())

    with open(f"{args.app_folder}/src/MHSAPULPNN.c", "w") as f:
        f.write(s)