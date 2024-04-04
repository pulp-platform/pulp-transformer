import numpy as np
import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import copy
from collections import OrderedDict
import sys
from mako.template import Template
import os
from templates_writer import print_template_multi_linear_l2_l1
from templates_writer import print_template_matmul_softmax_l2_l1
from templates_writer import print_template_matmul_l2_l1
import pandas as pd
import argparse 
from argparse import RawTextHelpFormatter
from tiling_creation import Tiling

def copy_files(sdk = 'gap_sdk',
				chip = 'GAP8v3',
				dma_parallelization = '8-cores'):
    ## copy backend and necessary files in the application folder
    os.system('rm -rf application')
    os.system('mkdir application')
    os.system('mkdir application/inc')
    os.system('mkdir application/src')
    os.system('cp ./layers/include/*  ./application/inc/')
    os.system('cp ./layers/src/*  ./application/src/')
    os.system('cp ./layers/Makefile  ./application/')
    os.system('cp ./templates/dory.c  ./application/src/')
    os.system('cp ./templates/dory.h  ./application/inc/')
    tk = OrderedDict([])
    tk['sdk'] = sdk
    root = '/'.join(os.getcwd().split('/')[:])
    tmpl = Template(filename=root + "/templates/dory.h")
    s = tmpl.render(**tk)
    save_string = './application/inc/dory.h'
    with open(save_string, "w") as f:
        f.write(s)
    tk = OrderedDict([])
    tk['sdk'] = sdk
    tmpl = Template(filename=root+"/templates/mchan_test.h")
    s = tmpl.render(**tk)
    save_string = './application/inc/mchan_test.h'
    with open(save_string, "w") as f:
        f.write(s)
    tk = OrderedDict([])
    tk['chip'] = chip
    tk['dma_parallelization'] = dma_parallelization
    tmpl = Template(filename=root+"/templates/dory.c")
    s = tmpl.render(**tk)
    save_string = './application/src/dory.c'
    with open(save_string, "w") as f:
        f.write(s)


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
        s = repr(x.flatten()).replace(", dtype=float32","").replace("dtype=float32","").replace("array([", "").replace("]", "").replace("[", "").replace(")", "").replace(",\n      dtype=int8)", "").replace(", dtype=uint8", "").replace(",\n      dtype=uint8)", "").replace(",\n      dtype=uint8", "").replace(",\n      dtype=int8", "").replace(", dtype=int8", "").replace(", dtype=int8)", "").replace(", dtype=int8)", "").replace(", dtype=uint8)", "")

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

def print_attention_template(
    dim_seq,
    heads,
    projections,
    embedding,
    x, 
    query_layer,
    key_layer, 
    value_layer, 
    attention_scores, 
    context_layer,
    attention_output, 
    q_weights, 
    k_weights, 
    v_weights,
    out_weights):
    # Generate the Layer management c file.
    tk = OrderedDict([])
    tk['dim_sequence'] = dim_seq
    tk['dim_embedding'] = embedding
    tk['projections'] = projections
    tk['heads'] = heads
    tk['input_content'] = print_test_vector(x.detach().numpy(), 'char')
    tk['W_q_content'] = print_test_vector(q_weights.detach().numpy(), 'char')
    tk['W_k_content'] = print_test_vector(k_weights.detach().numpy(), 'char')
    tk['W_v_content'] = print_test_vector(v_weights.detach().numpy(), 'char')
    tk['W_out_content'] = print_test_vector(out_weights.detach().numpy(), 'char')
    tk['qq_content'] = print_test_vector(query_layer.detach().numpy(), 'char')
    tk['kk_content'] = print_test_vector(key_layer.detach().numpy(), 'char')
    tk['vv_content'] = print_test_vector(value_layer.detach().numpy(), 'char')
    tk['matmul1_content'] = print_test_vector(attention_scores.detach().numpy(), 'char')
    tk['matmul2_content'] = print_test_vector(context_layer.detach().numpy(), 'char')
    tk['out_f_content'] = print_test_vector(attention_output.detach().numpy(), 'char')
    tk['check'] = False
    tk['check_byte'] = False
    tmpl = Template(filename="./test_layer_attention.c")
    s = tmpl.render(
        TEST=False,
        VERBOSE=False,
        ULTRA_VERBOSE=False,
        PULP_TEST=True,
        **tk)
    save_string = './application/src/main.c'
    with open(save_string, "w") as f:
        f.write(s)


def clip8(conv, bits):
    conv[conv >= +(2**(bits) - 1)] = +(2**(bits) - 1)
    conv[conv <= -(2**(bits))] = -(2**(bits))
    return conv


class Attention(nn.Module):
    def __init__(self, dim_seq, heads, projections, embedding):
        super(Attention, self).__init__()
        self.num_attention_heads = heads
        self.attention_head_size = projections
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(embedding, self.all_head_size)
        self.key = Linear(embedding, self.all_head_size)
        self.value = Linear(embedding, self.all_head_size)

        self.out = Linear(self.all_head_size, embedding)
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        mixed_query_layer = clip8(mixed_query_layer, 7)
        mixed_key_layer = clip8(mixed_key_layer, 7)
        mixed_value_layer = clip8(mixed_value_layer, 7)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = clip8(attention_scores, 7)
        #attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        #attention_probs = self.softmax(attention_scores)
        attention_probs = attention_scores
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = clip8(context_layer, 7)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = clip8(attention_output, 7)
        return query_layer, key_layer, value_layer, attention_scores, context_layer, attention_output

def attention_tests_generator(dim_seq, heads, projections, embedding, bits):
    # activations
    x = torch.Tensor(1, dim_seq, embedding).uniform_(0, (2**(bits + 1)))
    x[x > (2**(bits-1) - 1)] = 0
    x = torch.round(x)
    attention_layer = Attention(dim_seq, heads, projections, embedding)
    #weights = net[0].weight.data
    attention_layer.query.weight.data.random_(-(2**(bits - 1)), (2**(bits - 1)))
    attention_layer.key.weight.data.random_(-(2**(bits - 1)), (2**(bits - 1)))
    attention_layer.value.weight.data.random_(-(2**(bits - 1)), (2**(bits - 1)))
    attention_layer.out.weight.data.random_(-(2**(bits - 1)), (2**(bits - 1)))
    attention_layer.query.bias.data.uniform_(0, 0)
    attention_layer.key.bias.data.uniform_(0, 0)
    attention_layer.value.bias.data.uniform_(0, 0)
    attention_layer.out.bias.data.uniform_(0, 0)
    query_layer, key_layer, value_layer, attention_scores, context_layer, attention_output = attention_layer(x)
    return x, query_layer, key_layer, value_layer, attention_scores, context_layer, attention_output, attention_layer.query.weight.data, \
    attention_layer.key.weight.data, attention_layer.value.weight.data, attention_layer.out.weight.data


def main():
    """
    Create an instance of ONNX_management.
    If it is a test, create the onnx model.
    Then, extract infos and generate all the application folder.
    Give the folder on which you have 1 onnx file and the intermediate results
    """
    # ch_in, h, w, ch_out
    df = pd.read_excel('../Performance exploration.xlsx', sheet_name = 'Exploration_matmul')
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--layer',  type=int, default = 0)
    args = parser.parse_args()
    #layer = 11
    row = df[df['Test n.'] == args.layer]
    dim_seq = 10
    heads = 8
    projections = 32
    embedding = 64
    BitA = 8
    BitW = 8
    BitO = 8
    l2_tile_dimension_embedding = embedding
    l2_tile_dimension_proj = projections
    l2_tile_dimension_sequence = dim_seq
    l2_tile_dimension_heads = heads
    l2_tile_dimension_sequence_x1 = dim_seq
    l2_tile_dimension_heads_x1 = heads
    l2_tile_dimension_sequence_x2 = dim_seq
    l2_tile_dimension_heads_x2 = heads
    ds_x = 8; ds_y = 8; ds_W = 8
    ds_x1 = 8; ds_y = 8; ds_x2 = 8
    relu = 0; BN = 0; act_dim_bit = 32
    out_mul = 1.0; out_shift = 1.0
    l2_tile_dimension_seq_internal = dim_seq
    l2_tile_dimension_seq_x1 = dim_seq
    l2_tile_dimension_proj_x2 = projections
    print(f"L2 memory occupation of test: {dim_seq*embedding} + {embedding*projections*heads*3} + {dim_seq*projections*heads*8} + {dim_seq*dim_seq*heads*2} = {dim_seq*embedding + embedding*projections*heads*3 + dim_seq*projections*heads*8 + dim_seq*dim_seq*heads*2} ")
    mem = max(dim_seq*embedding + embedding*projections*heads + dim_seq*projections*heads, dim_seq*projections*heads*2 + dim_seq*dim_seq*heads)
    print(f"L1 memory occupation of test if not tiled: {mem} ")
    #### LINEAR LAYERS L1 PARAMETERS ####
    l1_tile_dimension_embedding = embedding
    l1_tile_dimension_sequence = dim_seq
    l1_tile_dimension_proj = projections
    l1_tile_dimension_heads = heads
    mem = max(l1_tile_dimension_sequence*embedding + embedding*l1_tile_dimension_proj*l1_tile_dimension_heads + l1_tile_dimension_sequence*l1_tile_dimension_proj*l1_tile_dimension_heads, dim_seq*projections*heads*2 + dim_seq*dim_seq*heads)
    print(f"L1 memory occupation after tiling linear: {mem} ")
    #### MATMUL SOFTMAX L1 PARAMETERS ####
    l1_tile_dimension_sequence_x1 = dim_seq
    l1_tile_dimension_heads_x1 = heads
    l1_tile_dimension_sequence_x2 = dim_seq
    l1_tile_dimension_heads_x2 = heads
    l1_tile_dimension_proj_matsoft = projections
    #### MATMUL L1 PARAMETERS ####
    l1_tile_dimension_seq_internal = row['Seq'].values[0]
    l1_tile_dimension_seq_x1 = row['Seq'].values[0]
    l1_tile_dimension_proj_x2 = row['Proj'].values[0]
    l1_tile_dimension_heads_x1_mat = row['Heads'].values[0]
    l1_tile_dimension_heads_x2_mat = row['Heads'].values[0]
    torch.manual_seed(3)
    import random
    x, query_layer, key_layer, value_layer, attention_scores, context_layer, attention_output, \
    q_weights, k_weights, v_weights, o_weights = attention_tests_generator(l2_tile_dimension_sequence, \
    	l2_tile_dimension_heads, l2_tile_dimension_proj, l2_tile_dimension_embedding,3)
    copy_files()
    print_attention_template(l2_tile_dimension_sequence, l2_tile_dimension_heads, l2_tile_dimension_proj, l2_tile_dimension_embedding, \
    	x, query_layer, key_layer, value_layer, attention_scores, context_layer, attention_output, q_weights, k_weights, v_weights, o_weights)
    tiling_test = 0
    if tiling_test == 1:
        print_template_multi_linear_l2_l1(l2_tile_dimension_embedding,
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
                             name = 'layer_attention_q_input_reuse',
                             ultra_verbose = True,
                             optional = 'input_reuse',
                             l1_buffer_dimension = 44000,
                             platform = 'GAP8',
                             chip = 'GAP8v3',
                             sdk = 'gap_sdk',
                             dma_parallelization = '8-cores')
        print_template_multi_linear_l2_l1(l2_tile_dimension_embedding,
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
                             name = 'layer_attention_k_input_reuse',
                             ultra_verbose = True,
                             optional = 'input_reuse',
                             l1_buffer_dimension = 44000,
                             platform = 'GAP8',
                             chip = 'GAP8v3',
                             sdk = 'gap_sdk',
                             dma_parallelization = '8-cores')
        print_template_multi_linear_l2_l1(l2_tile_dimension_embedding,
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
                             name = 'layer_attention_v_weight_reuse',
                             ultra_verbose = True,
                             optional = 'weight_reuse',
                             l1_buffer_dimension = 44000,
                             platform = 'GAP8',
                             chip = 'GAP8v3',
                             sdk = 'gap_sdk',
                             dma_parallelization = '8-cores')
        print_template_matmul_softmax_l2_l1(
                             l2_tile_dimension_proj,
                             l2_tile_dimension_sequence_x1,
                             l2_tile_dimension_heads_x1,
                             l2_tile_dimension_sequence_x2,
                             l2_tile_dimension_heads_x2,
                             l1_tile_dimension_proj_matsoft,
                             l1_tile_dimension_sequence_x1,
                             l1_tile_dimension_heads_x1,
                             l1_tile_dimension_sequence_x2,
                             l1_tile_dimension_heads_x2, # equal to l1_tile_dimension_heads_x1
                             ds_x1, ds_y, ds_x2,
                             relu, BN, act_dim_bit,
                             out_mul, out_shift,
                             name='layer_matmul_softmax',
                             ultra_verbose=True,
                             optional='None',
                             l1_buffer_dimension=44000,
                             platform='GAP8',
                             chip='GAP8v3',
                             sdk = 'gap_sdk',
                             dma_parallelization='8-cores')
        print_template_matmul_l2_l1(
                             l2_tile_dimension_seq_internal,
                             l2_tile_dimension_seq_x1,
                             l2_tile_dimension_heads_x1,
                             l2_tile_dimension_heads_x2,
                             l2_tile_dimension_proj_x2,
                             l1_tile_dimension_seq_internal,
                             l1_tile_dimension_seq_x1,
                             l1_tile_dimension_heads_x1_mat,
                             l1_tile_dimension_heads_x2_mat,
                             l1_tile_dimension_proj_x2,
                             ds_x1, ds_y, ds_x2,
                             relu, BN, act_dim_bit,
                             out_mul, out_shift,
                             name='layer_matmul',
                             ultra_verbose=True,
                             optional='None',
                             l1_buffer_dimension=44000,
                             platform='GAP8',
                             chip='GAP8v3',
                             sdk = 'gap_sdk',
                             dma_parallelization='8-cores')
    else:
        tiler = Tiling(44000, 400000, 'GAP8', 'GAP8v3', 8, 8, 8, 0, 0, 32, 'layer_attention_q_input_reuse', 'input_reuse', 'gap_sdk', '8-cores')
        tiler.get_tiling_multi_linear(l2_tile_dimension_embedding, l2_tile_dimension_proj, l2_tile_dimension_sequence, l2_tile_dimension_heads)
        tiler = Tiling(44000, 400000, 'GAP8', 'GAP8v3', 8, 8, 8, 0, 0, 32, 'layer_attention_k_input_reuse', 'input_reuse', 'gap_sdk', '8-cores')
        tiler.get_tiling_multi_linear(l2_tile_dimension_embedding, l2_tile_dimension_proj, l2_tile_dimension_sequence, l2_tile_dimension_heads)
        tiler = Tiling(44000, 400000, 'GAP8', 'GAP8v3', 8, 8, 8, 0, 0, 32, 'layer_attention_v_weight_reuse', 'weight_reuse', 'gap_sdk', '8-cores')
        tiler.get_tiling_multi_linear(l2_tile_dimension_embedding, l2_tile_dimension_proj, l2_tile_dimension_sequence, l2_tile_dimension_heads)
        tiler = Tiling(44000, 400000, 'GAP8', 'GAP8v3', 8, 8, 8, 0, 0, 32, 'layer_matmul_softmax', 'None', 'gap_sdk', '8-cores')
        tiler.get_tiling_matmul_softmax(l2_tile_dimension_proj, l2_tile_dimension_sequence_x1, l2_tile_dimension_heads_x1, l2_tile_dimension_sequence_x2, l2_tile_dimension_heads_x2)
        tiler = Tiling(44000, 400000, 'GAP8', 'GAP8v3', 8, 8, 8, 0, 0, 32, 'layer_matmul', 'None', 'gap_sdk', '8-cores')
        tiler.get_tiling_matmul(l2_tile_dimension_seq_internal, l2_tile_dimension_seq_x1, l2_tile_dimension_heads_x1, l2_tile_dimension_heads_x2, l2_tile_dimension_proj_x2)
        tiler = Tiling(44000, 400000, 'GAP8', 'GAP8v3', 8, 8, 8, 0, 0, 32, 'layer_attention_out', 'out', 'gap_sdk', '8-cores')
        tiler.get_tiling_multi_linear(l2_tile_dimension_embedding, l2_tile_dimension_proj, l2_tile_dimension_sequence, l2_tile_dimension_heads)


if __name__ == '__main__':
    main()

