#
# template.py
# Alessio Burrello <alessio.burrello@unibo.it>
#
# Copyright (C) 2019-2020 University of Bologna
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from mako.template import Template
import re
from collections import OrderedDict
import numpy as np
import sys
import os

def print_test_vector(x, type_data):
    # Print the test vector in the c file.
    if type_data == 'char':
        try:
            np.set_printoptions(
                threshold=sys.maxsize,
                formatter={'int': lambda x: hex(np.uint8(x)) if (
                    x < 0) else hex(np.uint8(x)), }
            )
        except TypeError:
            np.set_printoptions(threshold=sys.maxsize)
        s = repr(x.flatten()).replace("array([", "").replace("]", "").replace("[", "").replace(")", "").replace(",\n      dtype=int8)", "").replace(", dtype=uint8", "").replace(",\n      dtype=uint8)", "").replace(",\n      dtype=uint8", "").replace(",\n      dtype=int8", "").replace(", dtype=int8", "").replace(", dtype=int8)", "").replace(", dtype=int8)", "").replace(", dtype=uint8)", "")

    elif type_data == 'int16_t':
        try:
            np.set_printoptions(
                threshold=sys.maxsize,
                formatter={'int': lambda x: hex(np.uint16(x)) if (
                    x < 0) else hex(np.int16(x)), }
            )
        except TypeError:
            np.set_printoptions(threshold=sys.maxsize)
        s = repr(x.flatten()).replace("array([", "").replace("]", "").replace("[", "").replace(",\n      dtype=int16)", "").replace(
            ", dtype=int16)", "").replace(", dtype=int16)", "").replace(", dtype=uint16)", "").replace(")", "")

    else:
        try:
            np.set_printoptions(
                threshold=sys.maxsize,
                formatter={'int': lambda x: hex(np.uint32(x)) if (
                    x < 0) else hex(np.int32(x)), }
            )
        except TypeError:
            np.set_printoptions(threshold=sys.maxsize)
        s = repr(x.flatten()).replace("array([", "").replace("]", "").replace("[", "").replace(
            ",\n      dtype=int32)", "").replace(", dtype=int32)", "").replace(", dtype=int32)", "").replace(", dtype=uint32)", "")
    return s

def print_file_list(x):
    # This function is used to generate a string with all input files.
    s = repr(x).replace("[", "").replace("]", "").replace("'", '"')
    return s

def print_template_multi_linear_l2_l1(
                         l2_tile_dimension_embedding,
                         l2_tile_dimension_proj,
                         l2_tile_dimension_sequence,
                         l2_tile_dimension_heads,
                         l1_tile_dimension_embedding,
                         l1_tile_dimension_heads,
                         l1_tile_dimension_proj,
                         l1_tile_dimension_sequence,
                         ds_x, ds_y, ds_W,
                         relu, BN, act_dim_bit,
                         out_mul, out_shift,
                         name='layer',
                         ultra_verbose=True,
                         optional='input_reuse',
                         l1_buffer_dimension=44000,
                         platform='GAP8',
                         chip='GAP8v2',
                         sdk = 'gap_sdk',
                         dma_parallelization='8-cores'):
    # Generate the Layer management c file.
    # add padding from "regular" tile where necessary
    l1_num_of_tiles_heads = l2_tile_dimension_heads // l1_tile_dimension_heads + int((l2_tile_dimension_heads % l1_tile_dimension_heads) != 0)
    l1_num_of_tiles_proj = l2_tile_dimension_proj // l1_tile_dimension_proj + int((l2_tile_dimension_proj % l1_tile_dimension_proj) != 0)
    l1_num_of_tiles_seq = l2_tile_dimension_sequence // l1_tile_dimension_sequence + int((l2_tile_dimension_sequence % l1_tile_dimension_sequence) != 0)
    name_layer = re.sub(r'\W', '', name).replace("hex", "").replace(".", "").replace("_weights", "")
    tk = OrderedDict([])
    tk['func_name'] = name
    tk['FLAG_BATCHNORM'] = BN
    tk['act_dim_bit'] = act_dim_bit
    tk['chip'] = chip
    tk['sdk'] = sdk
    tk['dma_parallelization'] = dma_parallelization
    tk['platform'] = platform
    tk['x_data_size_bit'] = ds_x
    tk['y_data_size_bit'] = ds_y
    tk['W_data_size_bit'] = ds_W
    tk['x_tile_size'] = int(math.ceil(ds_x * l1_tile_dimension_sequence * l1_tile_dimension_embedding / 8.0))
    tk['y_tile_size'] = int(math.ceil(ds_y * l1_tile_dimension_sequence * l1_tile_dimension_heads * l1_tile_dimension_proj  / 8.0))
    tk['W_tile_size'] = int(math.ceil(ds_W * l1_tile_dimension_proj * l1_tile_dimension_heads * l1_tile_dimension_embedding / 8.0))
    tk['l1_tile_dimension_embedding'] = l1_tile_dimension_embedding
    tk['l1_tile_dimension_heads'] = l1_tile_dimension_heads
    tk['l1_tile_dimension_proj'] = l1_tile_dimension_proj
    tk['l1_tile_dimension_seq'] = l1_tile_dimension_sequence
    tk['l1_num_of_tiles_heads'] = l1_num_of_tiles_heads
    tk['l1_num_of_tiles_proj'] = l1_num_of_tiles_proj
    tk['l1_num_of_tiles_seq'] = l1_num_of_tiles_seq
    tk['l1_tile_dimension_seq_last'] = l2_tile_dimension_sequence % l1_tile_dimension_sequence if (l2_tile_dimension_sequence % l1_tile_dimension_sequence > 0) else l1_tile_dimension_sequence
    tk['l1_tile_dimension_proj_last'] = l2_tile_dimension_proj % l1_tile_dimension_proj if (l2_tile_dimension_proj % l1_tile_dimension_proj > 0) else l1_tile_dimension_proj
    tk['l1_tile_dimension_heads_last'] = l2_tile_dimension_heads % l1_tile_dimension_heads if (l2_tile_dimension_heads % l1_tile_dimension_heads > 0) else l1_tile_dimension_heads
    tk['l2_tile_dimension_embedding'] = l2_tile_dimension_embedding
    tk['l2_tile_dimension_proj'] = l2_tile_dimension_proj
    tk['l2_tile_dimension_sequence'] = l2_tile_dimension_sequence
    tk['l2_tile_dimension_heads'] = l2_tile_dimension_heads
    if l1_tile_dimension_sequence == l2_tile_dimension_sequence and l1_tile_dimension_embedding == l2_tile_dimension_embedding:
        x_buffer_size = int(math.ceil(ds_x * l1_tile_dimension_sequence * l1_tile_dimension_embedding / 8.0))
    else:
        x_buffer_size = 2 * int(math.ceil(ds_x * l1_tile_dimension_sequence * l1_tile_dimension_embedding / 8.0))
    if l1_tile_dimension_sequence == l2_tile_dimension_sequence and l1_tile_dimension_heads == l2_tile_dimension_heads and l1_tile_dimension_proj == l2_tile_dimension_proj:
        y_buffer_size = int(math.ceil(ds_y * l1_tile_dimension_sequence * l1_tile_dimension_heads * l1_tile_dimension_proj  / 8.0))
    else:
        y_buffer_size = 2 * int(math.ceil(ds_y * l1_tile_dimension_sequence * l1_tile_dimension_heads * l1_tile_dimension_proj  / 8.0))
    if l1_tile_dimension_proj == l2_tile_dimension_proj and l1_tile_dimension_heads == l2_tile_dimension_heads and l1_tile_dimension_embedding == l2_tile_dimension_embedding:
        W_buffer_size = int(math.ceil(ds_W * l1_tile_dimension_proj * l1_tile_dimension_heads * l1_tile_dimension_embedding / 8.0))
    else:
        W_buffer_size = 2 * int(math.ceil(ds_W * l1_tile_dimension_proj * l1_tile_dimension_heads * l1_tile_dimension_embedding / 8.0))
    tk['l1_y_offset'] = x_buffer_size + 4
    tk['l1_W_offset'] = x_buffer_size + 4 + y_buffer_size + 4
    l = ""
    for k, v in tk.items():
        try:
            l += "// %s %d\n" % (k.ljust(30), v)
        except TypeError:
            try:
                l += "// %s %d\n" % (k.ljust(30), v[0])
            except TypeError:
                l += "// %s %s\n" % (k.ljust(30), v)
    buffer_l1_all = W_buffer_size + x_buffer_size + y_buffer_size + 40
    tk['buffer_l1_all'] = buffer_l1_all
    root = '/'.join(os.getcwd().split('/')[:])
    tmpl = Template(filename=root + "/templates/layer_templates/multi_linear_layer_template.c")
    s = tmpl.render(VERBOSE=False,ULTRA_VERBOSE=ultra_verbose,PULP_TEST=True,verbose_log=l,**tk)
    save_string = './application/src/' + name_layer + '.c'
    with open(save_string, "w") as f:
        f.write(s)
    root = '/'.join(os.getcwd().split('/')[:])
    tmpl = Template(filename=root + "/templates/layer_templates/layer_template_h.h")
    s = tmpl.render(VERBOSE=False,ULTRA_VERBOSE=ultra_verbose,PULP_TEST=True,verbose_log=l,**tk)
    save_string = './application/inc/' + name_layer + '.h'
    with open(save_string, "w") as f:
        f.write(s)

def print_template_matmul_softmax_l2_l1(
                         l2_tile_dimension_proj,
                         l2_tile_dimension_sequence_x1,
                         l2_tile_dimension_heads_x1,
                         l2_tile_dimension_sequence_x2,
                         l2_tile_dimension_heads_x2,
                         l1_tile_dimension_proj,
                         l1_tile_dimension_sequence_x1,
                         l1_tile_dimension_heads_x1,
                         l1_tile_dimension_sequence_x2,
                         l1_tile_dimension_heads_x2, # equal to l1_tile_dimension_heads_x1
                         ds_x1, ds_y, ds_x2,
                         relu, BN, act_dim_bit,
                         out_mul, out_shift,
                         name='layer',
                         ultra_verbose=True,
                         optional='None',
                         l1_buffer_dimension=44000,
                         platform='GAP8',
                         chip='GAP8v2',
                         sdk = 'gap_sdk',
                         dma_parallelization='8-cores'):
    # Generate the Layer management c file.
    # add padding from "regular" tile where necessary
    l1_num_of_tiles_proj = l2_tile_dimension_proj // l1_tile_dimension_proj + int((l2_tile_dimension_proj % l1_tile_dimension_proj) != 0)
    l1_num_of_x1_tiles_heads = l2_tile_dimension_heads_x1 // l1_tile_dimension_heads_x1 + int((l2_tile_dimension_heads_x1 % l1_tile_dimension_heads_x1) != 0)
    l1_num_of_x1_tiles_seq = l2_tile_dimension_sequence_x1 // l1_tile_dimension_sequence_x1 + int((l2_tile_dimension_sequence_x1 % l1_tile_dimension_sequence_x1) != 0)
    l1_num_of_x2_tiles_heads = l2_tile_dimension_heads_x2 // l1_tile_dimension_heads_x2 + int((l2_tile_dimension_heads_x2 % l1_tile_dimension_heads_x2) != 0)
    l1_num_of_x2_tiles_seq = l2_tile_dimension_sequence_x2 // l1_tile_dimension_sequence_x2 + int((l2_tile_dimension_sequence_x2 % l1_tile_dimension_sequence_x2) != 0)
    name_layer = re.sub(r'\W', '', name).replace("hex", "").replace(".", "").replace("_weights", "")
    tk = OrderedDict([])
    tk['func_name'] = name
    tk['FLAG_BATCHNORM'] = BN
    tk['act_dim_bit'] = act_dim_bit
    tk['chip'] = chip
    tk['sdk'] = sdk
    tk['dma_parallelization'] = dma_parallelization
    tk['platform'] = platform
    tk['x1_data_size_bit'] = ds_x1
    tk['y_data_size_bit'] = ds_y
    tk['x2_data_size_bit'] = ds_x2
    tk['x1_tile_size'] = int(math.ceil(ds_x1 * l1_tile_dimension_heads_x1 * l1_tile_dimension_sequence_x1 * l1_tile_dimension_proj / 8.0))
    tk['x2_tile_size'] = int(math.ceil(ds_x2 * l1_tile_dimension_heads_x2 * l1_tile_dimension_sequence_x2 * l1_tile_dimension_proj / 8.0))
    tk['y_tile_size'] = int(math.ceil(ds_y * l1_tile_dimension_sequence_x1 * l1_tile_dimension_heads_x1 * l1_tile_dimension_sequence_x2  / 8.0))
    tk['l1_tile_dimension_proj'] = l1_tile_dimension_proj
    tk['l1_tile_dimension_heads_x1'] = l1_tile_dimension_heads_x1
    tk['l1_tile_dimension_seq_x1'] = l1_tile_dimension_sequence_x1
    tk['l1_tile_dimension_heads_x2'] = l1_tile_dimension_heads_x2
    tk['l1_tile_dimension_seq_x2'] = l1_tile_dimension_sequence_x2
    tk['l1_tile_dimension_heads_y'] = l1_tile_dimension_heads_x2
    tk['l1_tile_dimension_seq_y_ext'] = l1_tile_dimension_sequence_x1
    tk['l1_tile_dimension_seq_y_int'] = l1_tile_dimension_sequence_x2
    tk['l1_num_of_tiles_proj'] = l1_num_of_tiles_proj
    tk['l1_num_of_x1_tiles_heads'] = l1_num_of_x1_tiles_heads
    tk['l1_num_of_x1_tiles_seq'] = l1_num_of_x1_tiles_seq
    tk['l1_num_of_x2_tiles_heads'] = l1_num_of_x2_tiles_heads
    tk['l1_num_of_x2_tiles_seq'] = l1_num_of_x2_tiles_seq
    tk['l1_num_of_y_tiles_heads'] = l1_num_of_x1_tiles_heads
    tk['l1_num_of_y_tiles_seq_ext'] = l1_num_of_x1_tiles_seq
    tk['l1_num_of_y_tiles_seq_int'] = l1_num_of_x2_tiles_seq
    tk['l1_tile_dimension_proj_last'] = l2_tile_dimension_proj % l1_tile_dimension_proj if (l2_tile_dimension_proj % l1_tile_dimension_proj > 0) else l1_tile_dimension_proj
    tk['l1_tile_dimension_seq_x1_last'] = l2_tile_dimension_sequence_x1 % l1_tile_dimension_sequence_x1 if (l2_tile_dimension_sequence_x1 % l1_tile_dimension_sequence_x1 > 0) else l1_tile_dimension_sequence_x1
    tk['l1_tile_dimension_heads_x1_last'] = l2_tile_dimension_heads_x1 % l1_tile_dimension_heads_x1 if (l2_tile_dimension_heads_x1 % l1_tile_dimension_heads_x1 > 0) else l1_tile_dimension_heads_x1
    tk['l1_tile_dimension_seq_x2_last'] = l2_tile_dimension_sequence_x2 % l1_tile_dimension_sequence_x2 if (l2_tile_dimension_sequence_x2 % l1_tile_dimension_sequence_x2 > 0) else l1_tile_dimension_sequence_x2
    tk['l1_tile_dimension_heads_x2_last'] = l2_tile_dimension_heads_x2 % l1_tile_dimension_heads_x2 if (l2_tile_dimension_heads_x2 % l1_tile_dimension_heads_x2 > 0) else l1_tile_dimension_heads_x2
    tk['l2_tile_dimension_proj'] = l2_tile_dimension_proj
    tk['l2_tile_dimension_sequence_x1'] = l2_tile_dimension_sequence_x1
    tk['l2_tile_dimension_heads_x1'] = l2_tile_dimension_heads_x1
    tk['l2_tile_dimension_sequence_x2'] = l2_tile_dimension_sequence_x2
    tk['l2_tile_dimension_heads_x2'] = l2_tile_dimension_heads_x2
    if l1_tile_dimension_sequence_x1 == l2_tile_dimension_sequence_x1 and l1_tile_dimension_heads_x1 == l2_tile_dimension_heads_x1 and l2_tile_dimension_proj == l1_tile_dimension_proj:
        x1_buffer_size = int(math.ceil(ds_x1 * l1_tile_dimension_heads_x1 * l1_tile_dimension_sequence_x1 * l1_tile_dimension_proj / 8.0))
    else:
        x1_buffer_size = 2 * int(math.ceil(ds_x1 * l1_tile_dimension_heads_x1 * l1_tile_dimension_sequence_x1 * l1_tile_dimension_proj / 8.0))
    if l1_tile_dimension_sequence_x2 == l2_tile_dimension_sequence_x2 and l1_tile_dimension_heads_x1 == l2_tile_dimension_heads_x1 and l1_tile_dimension_sequence_x1 == l2_tile_dimension_sequence_x1:
        y_buffer_size = int(math.ceil(ds_y * l1_tile_dimension_sequence_x2 * l1_tile_dimension_heads_x1 * l1_tile_dimension_sequence_x1  / 8.0))
    else:
        y_buffer_size = 2 * int(math.ceil(ds_y * l1_tile_dimension_sequence_x2 * l1_tile_dimension_heads_x1 * l1_tile_dimension_sequence_x1 / 8.0))
    if l1_tile_dimension_sequence_x2 == l2_tile_dimension_sequence_x2 and l1_tile_dimension_heads_x2 == l2_tile_dimension_heads_x2 and l2_tile_dimension_proj == l1_tile_dimension_proj:
        x2_buffer_size = int(math.ceil(ds_x2 * l1_tile_dimension_heads_x2 * l1_tile_dimension_sequence_x2 * l1_tile_dimension_proj / 8.0))
    else:
        x2_buffer_size = 2 * int(math.ceil(ds_x2 * l1_tile_dimension_heads_x2 * l1_tile_dimension_sequence_x2 * l1_tile_dimension_proj / 8.0))
    tk['l1_y_offset'] = x1_buffer_size + 4
    tk['l1_x2_offset'] = x1_buffer_size + 4 + y_buffer_size + 4
    l = ""
    for k, v in tk.items():
        try:
            l += "// %s %d\n" % (k.ljust(30), v)
        except TypeError:
            try:
                l += "// %s %d\n" % (k.ljust(30), v[0])
            except TypeError:
                l += "// %s %s\n" % (k.ljust(30), v)
    buffer_l1_all = x1_buffer_size + x2_buffer_size + y_buffer_size + 40
    tk['buffer_l1_all'] = buffer_l1_all
    root = '/'.join(os.getcwd().split('/')[:])
    tmpl = Template(filename=root + "/templates/layer_templates/matmul_softmax_layer_template.c")
    s = tmpl.render(VERBOSE=False,ULTRA_VERBOSE=ultra_verbose,PULP_TEST=True,verbose_log=l,**tk)
    save_string = './application/src/' + name_layer + '.c'
    with open(save_string, "w") as f:
        f.write(s)
    root = '/'.join(os.getcwd().split('/')[:])
    tmpl = Template(filename=root + "/templates/layer_templates/layer_template_h.h")
    s = tmpl.render(VERBOSE=False,ULTRA_VERBOSE=ultra_verbose,PULP_TEST=True,verbose_log=l,**tk)
    save_string = './application/inc/' + name_layer + '.h'
    with open(save_string, "w") as f:
        f.write(s)

def print_template_matmul_l2_l1(
                         l2_tile_dimension_seq_internal,
                         l2_tile_dimension_seq_x1,
                         l2_tile_dimension_heads_x1,
                         l2_tile_dimension_heads_x2,
                         l2_tile_dimension_proj_x2,
                         l1_tile_dimension_seq_internal,
                         l1_tile_dimension_seq_x1,
                         l1_tile_dimension_heads_x1,
                         l1_tile_dimension_heads_x2,
                         l1_tile_dimension_proj_x2,
                         ds_x1, ds_y, ds_x2,
                         relu, BN, act_dim_bit,
                         out_mul, out_shift,
                         name='layer',
                         ultra_verbose=True,
                         optional='None',
                         l1_buffer_dimension=44000,
                         platform='GAP8',
                         chip='GAP8v2',
                         sdk = 'gap_sdk',
                         dma_parallelization='8-cores'):
    # Generate the Layer management c file.
    # add padding from "regular" tile where necessary
    l1_num_of_x1_tiles_heads = l2_tile_dimension_heads_x1 // l1_tile_dimension_heads_x1 + int((l2_tile_dimension_heads_x1 % l1_tile_dimension_heads_x1) != 0)
    l1_num_of_x1_tiles_seq = l2_tile_dimension_seq_x1 // l1_tile_dimension_seq_x1 + int((l2_tile_dimension_seq_x1 % l1_tile_dimension_seq_x1) != 0)
    l1_num_of_x2_tiles_heads = l2_tile_dimension_heads_x2 // l1_tile_dimension_heads_x2 + int((l2_tile_dimension_heads_x2 % l1_tile_dimension_heads_x2) != 0)
    l1_num_of_x2_tiles_proj = l2_tile_dimension_proj_x2 // l1_tile_dimension_proj_x2 + int((l2_tile_dimension_proj_x2 % l1_tile_dimension_proj_x2) != 0)
    l1_num_of_seq_internal_tiles = l2_tile_dimension_seq_internal // l1_tile_dimension_seq_internal + int((l2_tile_dimension_seq_internal % l1_tile_dimension_seq_internal) != 0)
    name_layer = re.sub(r'\W', '', name).replace("hex", "").replace(".", "").replace("_weights", "")
    tk = OrderedDict([])
    tk['func_name'] = name
    tk['FLAG_BATCHNORM'] = BN
    tk['act_dim_bit'] = act_dim_bit
    tk['chip'] = chip
    tk['sdk'] = sdk
    tk['dma_parallelization'] = dma_parallelization
    tk['platform'] = platform
    tk['x1_data_size_bit'] = ds_x1
    tk['y_data_size_bit'] = ds_y
    tk['x2_data_size_bit'] = ds_x2
    tk['x1_tile_size'] = int(math.ceil(ds_x1 * l1_tile_dimension_heads_x1 * l1_tile_dimension_seq_x1 * l1_tile_dimension_seq_internal / 8.0))
    tk['x2_tile_size'] = int(math.ceil(ds_x2 * l1_tile_dimension_heads_x2 * l1_tile_dimension_proj_x2 * l1_tile_dimension_seq_internal / 8.0))
    tk['y_tile_size'] = int(math.ceil(ds_y * l1_tile_dimension_seq_x1 * l1_tile_dimension_proj_x2 * l1_tile_dimension_heads_x2 / 8.0))
    
    tk['l1_tile_dimension_seq_internal'] = l1_tile_dimension_seq_internal
    tk['l1_tile_dimension_seq_x1'] = l1_tile_dimension_seq_x1
    tk['l1_tile_dimension_heads_x1'] = l1_tile_dimension_heads_x1
    tk['l1_tile_dimension_heads_x2'] = l1_tile_dimension_heads_x2
    tk['l1_tile_dimension_proj_x2'] = l1_tile_dimension_proj_x2
    
    tk['l1_tile_dimension_heads_y'] = l1_tile_dimension_heads_x2
    tk['l1_tile_dimension_seq_y'] = l1_tile_dimension_seq_x1
    tk['l1_tile_dimension_proj_y'] = l1_tile_dimension_proj_x2
    
    tk['l2_tile_dimension_seq_internal'] = l2_tile_dimension_seq_internal
    tk['l2_tile_dimension_seq_x1'] = l2_tile_dimension_seq_x1
    tk['l2_tile_dimension_heads_x1'] = l2_tile_dimension_heads_x1
    tk['l2_tile_dimension_heads_x2'] = l2_tile_dimension_heads_x2
    tk['l2_tile_dimension_proj_x2'] = l2_tile_dimension_proj_x2
    
    tk['l1_num_of_seq_internal_tiles'] = l1_num_of_seq_internal_tiles
    tk['l1_num_of_x1_tiles_heads'] = l1_num_of_x1_tiles_heads
    tk['l1_num_of_x1_tiles_seq'] = l1_num_of_x1_tiles_seq
    tk['l1_num_of_x2_tiles_heads'] = l1_num_of_x2_tiles_heads
    tk['l1_num_of_x2_tiles_proj'] = l1_num_of_x2_tiles_proj
    tk['l1_num_of_y_tiles_heads'] = l1_num_of_x1_tiles_heads
    tk['l1_num_of_y_tiles_seq'] = l1_num_of_x1_tiles_seq
    tk['l1_num_of_y_tiles_proj'] = l1_num_of_x2_tiles_proj
    tk['l1_tile_dimension_seq_internal_last'] = l2_tile_dimension_seq_internal % l1_tile_dimension_seq_internal if (l2_tile_dimension_seq_internal % l1_tile_dimension_seq_internal > 0) else l1_tile_dimension_seq_internal
    tk['l1_tile_dimension_seq_x1_last'] = l2_tile_dimension_seq_x1 % l1_tile_dimension_seq_x1 if (l2_tile_dimension_seq_x1 % l1_tile_dimension_seq_x1 > 0) else l1_tile_dimension_seq_x1
    tk['l1_tile_dimension_heads_x1_last'] = l2_tile_dimension_heads_x1 % l1_tile_dimension_heads_x1 if (l2_tile_dimension_heads_x1 % l1_tile_dimension_heads_x1 > 0) else l1_tile_dimension_heads_x1
    tk['l1_tile_dimension_heads_x2_last'] = l2_tile_dimension_heads_x2 % l1_tile_dimension_heads_x2 if (l2_tile_dimension_heads_x2 % l1_tile_dimension_heads_x2 > 0) else l1_tile_dimension_heads_x2
    tk['l1_tile_dimension_proj_x2_last'] = l2_tile_dimension_proj_x2 % l1_tile_dimension_proj_x2 if (l2_tile_dimension_proj_x2 % l1_tile_dimension_proj_x2 > 0) else l1_tile_dimension_proj_x2
    if l1_tile_dimension_seq_x1 == l2_tile_dimension_seq_x1 and l1_tile_dimension_heads_x1 == l2_tile_dimension_heads_x1 and l2_tile_dimension_seq_internal == l1_tile_dimension_seq_internal:
        x1_buffer_size = int(math.ceil(ds_x1 * l1_tile_dimension_heads_x1 * l1_tile_dimension_seq_x1 * l1_tile_dimension_seq_internal / 8.0))
    else:
        x1_buffer_size = 2 * int(math.ceil(ds_x1 * l1_tile_dimension_heads_x1 * l1_tile_dimension_seq_x1 * l1_tile_dimension_seq_internal / 8.0))
    if l1_tile_dimension_proj_x2 == l2_tile_dimension_proj_x2 and l1_tile_dimension_heads_x2 == l2_tile_dimension_heads_x2 and l1_tile_dimension_seq_x1 == l2_tile_dimension_seq_x1:
        y_buffer_size = int(math.ceil(ds_y * l1_tile_dimension_seq_x1 * l1_tile_dimension_heads_x1 * l1_tile_dimension_proj_x2  / 8.0))
    else:
        y_buffer_size = 2 * int(math.ceil(ds_y * l1_tile_dimension_seq_x1 * l1_tile_dimension_heads_x1 * l1_tile_dimension_proj_x2 / 8.0))
    if l1_tile_dimension_seq_internal == l2_tile_dimension_seq_internal and l1_tile_dimension_heads_x2 == l2_tile_dimension_heads_x2 and l1_tile_dimension_proj_x2 == l2_tile_dimension_proj_x2:
        x2_buffer_size = int(math.ceil(ds_x2 * l1_tile_dimension_heads_x2 * l1_tile_dimension_seq_internal * l1_tile_dimension_proj_x2 / 8.0))
    else:
        x2_buffer_size = 2 * int(math.ceil(ds_x2 * l1_tile_dimension_heads_x2 * l1_tile_dimension_seq_internal * l1_tile_dimension_proj_x2 / 8.0))
    tk['l1_y_offset'] = x1_buffer_size + 4
    tk['l1_x2_offset'] = x1_buffer_size + 4 + y_buffer_size + 4
    l = ""
    for k, v in tk.items():
        try:
            l += "// %s %d\n" % (k.ljust(30), v)
        except TypeError:
            try:
                l += "// %s %d\n" % (k.ljust(30), v[0])
            except TypeError:
                l += "// %s %s\n" % (k.ljust(30), v)
    buffer_l1_all = x1_buffer_size + x2_buffer_size + y_buffer_size + 40
    tk['buffer_l1_all'] = buffer_l1_all
    root = '/'.join(os.getcwd().split('/')[:])
    tmpl = Template(filename=root + "/templates/layer_templates/matmul_layer_template.c")
    s = tmpl.render(VERBOSE=False,ULTRA_VERBOSE=ultra_verbose,PULP_TEST=True,verbose_log=l,**tk)
    save_string = './application/src/' + name_layer + '.c'
    with open(save_string, "w") as f:
        f.write(s)
    root = '/'.join(os.getcwd().split('/')[:])
    tmpl = Template(filename=root + "/templates/layer_templates/layer_template_h.h")
    s = tmpl.render(VERBOSE=False,ULTRA_VERBOSE=ultra_verbose,PULP_TEST=True,verbose_log=l,**tk)
    save_string = './application/inc/' + name_layer + '.h'
    with open(save_string, "w") as f:
        f.write(s)


