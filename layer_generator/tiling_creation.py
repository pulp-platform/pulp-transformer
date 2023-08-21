# -*- coding: future_fstrings -*-     # should work even without -*-
# tiling_creation.py
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
import math
import numpy as np
import torch
import torch.nn as nn

# constraint solver for optimization
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import solver_parameters_pb2

# template for output
from templates_writer import print_template_multi_linear_l2_l1
from templates_writer import print_template_matmul_softmax_l2_l1
from templates_writer import print_template_matmul_l2_l1
import logging
import os
import sys

class Tiling():
    # Class to generate the Tiling of the layer.
    def __init__(self,
            L1_buffer, 
            L2_buffer, 
            platform, 
            chip,
            ds_x, 
            ds_W, 
            ds_y, 
            relu, BN,
            act_dim_bit, 
            name, 
            optional, 
            sdk, 
            dma_parallelization):
        self.L1_buffer_size = L1_buffer
        self.L2_buffer_size = L2_buffer
        self.platform = platform
        self.chip = chip
        self.ds_x = ds_x
        self.ds_W = ds_W
        self.ds_y = ds_y
        self.relu = relu
        self.BN = BN
        self.act_dim_bit = act_dim_bit
        self.name = name
        self.optional = optional
        self.sdk = sdk
        self.dma_parallelization = dma_parallelization

    def get_tiling_multi_linear(self, 
                        l2_tile_dimension_embedding,
                        l2_tile_dimension_proj,
                        l2_tile_dimension_sequence,
                        l2_tile_dimension_heads):
        # This function generate the layer function to be included in the project for the addition operation.

        parameters = pywrapcp.Solver.DefaultSolverParameters()
        solver = pywrapcp.Solver("simple_CP", parameters)
        ds_x = self.ds_x
        ds_y = self.ds_y
        ds_W = self.ds_W
        # this is to renormalize all costs
        max_obj_value = sys.maxsize
        memory = ds_x * l2_tile_dimension_embedding * l2_tile_dimension_sequence + \
        ds_y * l2_tile_dimension_heads * l2_tile_dimension_proj * l2_tile_dimension_sequence + \
        ds_W * l2_tile_dimension_proj * l2_tile_dimension_heads * l2_tile_dimension_embedding
        if memory >= self.L2_buffer_size * 8:
            print("  Multi-Linear ERROR: no tiling from L3 supported. Exiting...")
            os._exit(0)            
        if memory <= self.L1_buffer_size * 8:
            db_x = 1
            db_y = 1
            db_W = 1
        else:
            db_x = 2
            db_y = 2
            db_W = 2
        # integer positive variables.
        tile_seq = solver.IntVar(1, l2_tile_dimension_sequence, 'tile_seq')
        tile_proj = solver.IntVar(1, l2_tile_dimension_proj, 'tile_proj')
        tile_heads = solver.IntVar(1, l2_tile_dimension_heads, 'tile_heads')

        # scaling is used to ensure datasize is integer
        solver.Add(tile_seq % 4 == 0)
        solver.Add(tile_proj % 4 == 0)  

        # CONSTRAINTS: managing of correct dimensions (no decimal h_out and any
        # type of rounding)
        solver.Add(ds_x * db_x * l2_tile_dimension_embedding * tile_seq + ds_y * db_y * tile_heads * tile_proj * tile_seq + ds_W * db_W * tile_proj * tile_heads * l2_tile_dimension_embedding <= self.L1_buffer_size * 8)
        # objective
        obj_expr = solver.IntVar(0, max_obj_value, "obj_expr")
        mem_total = ds_x * db_x * l2_tile_dimension_embedding * tile_seq + ds_y * db_y * tile_heads * tile_proj * tile_seq + ds_W * db_W * tile_proj * tile_heads * l2_tile_dimension_embedding
        solver.Add(obj_expr == 1 * mem_total
                   + 1 * tile_seq
                   + 1 * tile_proj
                   + 1 * tile_heads 
                   + 1000000 * ((tile_heads-1) % 8))
        objective = solver.Maximize(obj_expr, 1)

        decision_builder = solver.Phase([tile_seq, tile_proj, tile_heads],
                                        solver.CHOOSE_FIRST_UNBOUND,
                                        solver.ASSIGN_MIN_VALUE)

        # Create a solution collector.
        collector = solver.LastSolutionCollector()
        # Add the decision variables.
        collector.Add(tile_heads)
        collector.Add(tile_seq)
        collector.Add(tile_proj)
        # Add the objective.
        collector.AddObjective(obj_expr)

        solver.Solve(decision_builder, [objective, collector])
        if collector.SolutionCount() > 0:
            best_solution = collector.SolutionCount() - 1

            tile_heads = collector.Value(best_solution, tile_heads)
            tile_seq = collector.Value(best_solution, tile_seq)
            tile_proj = collector.Value(best_solution, tile_proj)

            input_dim = '[%dx%d]' % (l2_tile_dimension_sequence, l2_tile_dimension_embedding)
            W_dim = '[%dx%dx%d]' % (l2_tile_dimension_heads, l2_tile_dimension_proj, l2_tile_dimension_embedding)
            y_dim = '[%dx%dx%d]' % (l2_tile_dimension_heads, l2_tile_dimension_sequence, l2_tile_dimension_proj)
            input_dim_tile = '[%dx%d]' % (tile_seq, l2_tile_dimension_embedding)
            W_dim_tile = '[%dx%dx%d]' % (tile_heads, tile_seq, tile_proj)
            y_dim_tile = '[%dx%dx%d]' % (tile_heads, tile_seq, tile_proj)

            print("  Multi-Linear tiling:")
            print("    L2 size:".ljust(18) + "x: " +
                          input_dim.ljust(15) + "W: " + W_dim.ljust(15) + "y: " + y_dim.ljust(15))
            print("    L1 size:".ljust(18) + "x: " +
                          input_dim_tile.ljust(15) + "W: " + W_dim_tile.ljust(15) + "y: " + y_dim_tile.ljust(15))
            print_template_multi_linear_l2_l1(l2_tile_dimension_embedding,
                                 l2_tile_dimension_proj,
                                 l2_tile_dimension_sequence,
                                 l2_tile_dimension_heads,
                                 l2_tile_dimension_embedding,
                                 tile_heads,
                                 tile_proj,
                                 tile_seq,
                                 ds_x, ds_y, ds_W,
                                 self.relu, self.BN, self.act_dim_bit,
                                 1, 1,
                                 self.name,
                                 ultra_verbose = True,
                                 optional = self.optional,
                                 l1_buffer_dimension = self.L1_buffer_size,
                                 platform = self.platform,
                                 chip = self.chip,
                                 sdk = self.sdk,
                                 dma_parallelization = self.dma_parallelization)
            return None
        print("  ERROR: no tiling found. Exiting...")
        os._exit(0)
        return None

    def get_tiling_matmul_softmax(self, 
                             l2_tile_dimension_proj,
                             l2_tile_dimension_sequence_x1,
                             l2_tile_dimension_heads_x1,
                             l2_tile_dimension_sequence_x2,
                             l2_tile_dimension_heads_x2):
        # This function generate the layer function to be included in the project for the addition operation.

        parameters = pywrapcp.Solver.DefaultSolverParameters()
        solver = pywrapcp.Solver("simple_CP", parameters)
        ds_x = self.ds_x
        ds_y = self.ds_y
        ds_W = self.ds_W
        # this is to renormalize all costs
        max_obj_value = sys.maxsize
        memory = ds_x * l2_tile_dimension_heads_x1 * l2_tile_dimension_proj * l2_tile_dimension_sequence_x1 + \
        ds_W * l2_tile_dimension_heads_x2 * l2_tile_dimension_proj * l2_tile_dimension_sequence_x2 + \
        ds_y * l2_tile_dimension_sequence_x2 * l2_tile_dimension_sequence_x1 * l2_tile_dimension_heads_x1
        if memory >= self.L2_buffer_size * 8:
            print("  Matmul-Softmax ERROR: no tiling from L3 supported. Exiting...")
            os._exit(0)            
        if memory <= self.L1_buffer_size * 8:
            db_x = 1
            db_y = 1
            db_W = 1
        else:
            db_x = 2
            db_y = 2
            db_W = 2
        # integer positive variables.
        tile_heads = solver.IntVar(1, l2_tile_dimension_heads_x1, 'tile_heads')
        mem_total = ds_x * db_x * tile_heads * l2_tile_dimension_proj * l2_tile_dimension_sequence_x1 + \
                    ds_W * db_W * tile_heads * l2_tile_dimension_proj * l2_tile_dimension_sequence_x2 + \
                    ds_y * db_y * tile_heads * l2_tile_dimension_sequence_x1 * l2_tile_dimension_sequence_x2
        solver.Add(mem_total <= self.L1_buffer_size * 8)
        # objective
        obj_expr = solver.IntVar(0, max_obj_value, "obj_expr")
        solver.Add(obj_expr == 1 * mem_total
                   + 1 * tile_heads 
                   + 10000000000000 * ((tile_heads-1) % 8))
        objective = solver.Maximize(obj_expr, 1)

        decision_builder = solver.Phase([tile_heads],
                                        solver.CHOOSE_FIRST_UNBOUND,
                                        solver.ASSIGN_MIN_VALUE)

        # Create a solution collector.
        collector = solver.LastSolutionCollector()
        # Add the decision variables.
        collector.Add(tile_heads)
        # Add the objective.
        collector.AddObjective(obj_expr)

        solver.Solve(decision_builder, [objective, collector])
        if collector.SolutionCount() > 0:
            best_solution = collector.SolutionCount() - 1
            tile_heads = collector.Value(best_solution, tile_heads)
            input_dim = '[%dx%dx%d]' % (l2_tile_dimension_heads_x1, l2_tile_dimension_sequence_x1, l2_tile_dimension_proj)
            W_dim = '[%dx%dx%d]' % (l2_tile_dimension_heads_x2, l2_tile_dimension_sequence_x2, l2_tile_dimension_proj)
            y_dim = '[%dx%dx%d]' % (l2_tile_dimension_heads_x1, l2_tile_dimension_sequence_x1, l2_tile_dimension_sequence_x2)
            input_dim_tile = '[%dx%dx%d]' % (tile_heads, l2_tile_dimension_sequence_x1, l2_tile_dimension_proj)
            W_dim_tile = '[%dx%dx%d]' % (tile_heads, l2_tile_dimension_sequence_x2, l2_tile_dimension_proj)
            y_dim_tile = '[%dx%dx%d]' % (tile_heads, l2_tile_dimension_sequence_x1, l2_tile_dimension_sequence_x2)
            print("  Matmul-Softmax tiling:")
            print("    L2 size:".ljust(18) + "x1: " +
                          input_dim.ljust(14) + "x2: " + W_dim.ljust(14) + "y: " + y_dim.ljust(15))
            print("    L1 size:".ljust(18) + "x1: " +
                          input_dim_tile.ljust(14) + "x2: " + W_dim_tile.ljust(14) + "y: " + y_dim_tile.ljust(15))
            print_template_matmul_softmax_l2_l1(
                                 l2_tile_dimension_proj,
                                 l2_tile_dimension_sequence_x1,
                                 l2_tile_dimension_heads_x1,
                                 l2_tile_dimension_sequence_x2,
                                 l2_tile_dimension_heads_x2,
                                 l2_tile_dimension_proj,
                                 l2_tile_dimension_sequence_x1,
                                 tile_heads,
                                 l2_tile_dimension_sequence_x2,
                                 tile_heads,
                                 ds_x, ds_y, ds_W,
                                 self.relu, self.BN, self.act_dim_bit,
                                 1, 1,
                                 self.name,
                                 ultra_verbose = True,
                                 optional = self.optional,
                                 l1_buffer_dimension = self.L1_buffer_size,
                                 platform = self.platform,
                                 chip = self.chip,
                                 sdk = self.sdk,
                                 dma_parallelization = self.dma_parallelization)
            return None
        print("  ERROR: no tiling found. Exiting...")
        os._exit(0)
        return None

    def get_tiling_matmul(self, 
                             l2_tile_dimension_seq_internal,
                             l2_tile_dimension_seq_x1,
                             l2_tile_dimension_heads_x1,
                             l2_tile_dimension_heads_x2,
                             l2_tile_dimension_proj_x2):
        # This function generate the layer function to be included in the project for the addition operation.

        parameters = pywrapcp.Solver.DefaultSolverParameters()
        solver = pywrapcp.Solver("simple_CP", parameters)
        ds_x = self.ds_x
        ds_y = self.ds_y
        ds_W = self.ds_W
        # this is to renormalize all costs
        max_obj_value = sys.maxsize
        memory = ds_x * l2_tile_dimension_heads_x1 * l2_tile_dimension_seq_internal * l2_tile_dimension_seq_x1 + \
        ds_W * l2_tile_dimension_heads_x2 * l2_tile_dimension_proj_x2 * l2_tile_dimension_seq_internal + \
        ds_y * l2_tile_dimension_seq_x1 * l2_tile_dimension_heads_x2 * l2_tile_dimension_proj_x2
        if memory >= self.L2_buffer_size * 8:
            print("  Matmul ERROR: no tiling from L3 supported. Exiting...")
            os._exit(0)            
        if memory <= self.L1_buffer_size * 8:
            db_x = 1
            db_y = 1
            db_W = 1
        else:
            db_x = 2
            db_y = 2
            db_W = 2
        # integer positive variables.
        tile_heads = solver.IntVar(1, l2_tile_dimension_heads_x1, 'tile_heads')
        tile_proj = solver.IntVar(1, l2_tile_dimension_proj_x2, 'tile_proj')

        # scaling is used to ensure datasize is integer
        solver.Add(tile_proj % 4 == 0)  
        mem_total = ds_x * db_x * tile_heads * l2_tile_dimension_seq_internal * l2_tile_dimension_seq_x1 + \
                    ds_W * db_W * tile_heads * tile_proj * l2_tile_dimension_seq_internal + \
                    ds_y * db_y * tile_heads * tile_proj * l2_tile_dimension_seq_x1
        solver.Add(mem_total <= self.L1_buffer_size * 8)
        # objective
        obj_expr = solver.IntVar(0, max_obj_value, "obj_expr")
        solver.Add(obj_expr == 1 * mem_total
                   + 1 * tile_heads 
                   + 1 * tile_proj 
                   + 100000 * ((tile_heads-1) % 8))
        objective = solver.Maximize(obj_expr, 1)

        decision_builder = solver.Phase([tile_heads, tile_proj],
                                        solver.CHOOSE_FIRST_UNBOUND,
                                        solver.ASSIGN_MIN_VALUE)

        # Create a solution collector.
        collector = solver.LastSolutionCollector()
        # Add the decision variables.
        collector.Add(tile_heads)
        collector.Add(tile_proj)
        # Add the objective.
        collector.AddObjective(obj_expr)

        solver.Solve(decision_builder, [objective, collector])
        if collector.SolutionCount() > 0:
            best_solution = collector.SolutionCount() - 1
            tile_heads = collector.Value(best_solution, tile_heads)
            tile_proj = collector.Value(best_solution, tile_proj)
            input_dim = '[%dx%dx%d]' % (l2_tile_dimension_heads_x1, l2_tile_dimension_seq_x1, l2_tile_dimension_seq_internal)
            W_dim = '[%dx%dx%d]' % (l2_tile_dimension_heads_x2, l2_tile_dimension_proj_x2, l2_tile_dimension_seq_internal)
            y_dim = '[%dx%dx%d]' % (l2_tile_dimension_seq_x1, l2_tile_dimension_heads_x2, l2_tile_dimension_proj_x2)
            input_dim_tile = '[%dx%dx%d]' % (tile_heads, l2_tile_dimension_seq_x1, l2_tile_dimension_seq_internal)
            W_dim_tile = '[%dx%dx%d]' % (tile_heads, tile_proj, l2_tile_dimension_seq_internal)
            y_dim_tile = '[%dx%dx%d]' % (l2_tile_dimension_seq_x1, tile_heads, tile_proj)
            print("  Matmul tiling:")
            print("    L2 size:".ljust(18) + "x1: " +
                          input_dim.ljust(14) + "x2: " + W_dim.ljust(14) + "y: " + y_dim.ljust(15))
            print("    L1 size:".ljust(18) + "x1: " +
                          input_dim_tile.ljust(14) + "x2: " + W_dim_tile.ljust(14) + "y: " + y_dim_tile.ljust(15))
            print_template_matmul_l2_l1(
                                 l2_tile_dimension_seq_internal,
                                 l2_tile_dimension_seq_x1,
                                 l2_tile_dimension_heads_x1,
                                 l2_tile_dimension_heads_x2,
                                 l2_tile_dimension_proj_x2,
                                 l2_tile_dimension_seq_internal,
                                 l2_tile_dimension_seq_x1,
                                 tile_heads, tile_heads, tile_proj,
                                 ds_x, ds_y, ds_W,
                                 self.relu, self.BN, self.act_dim_bit,
                                 1, 1,
                                 self.name,
                                 ultra_verbose = True,
                                 optional = self.optional,
                                 l1_buffer_dimension = self.L1_buffer_size,
                                 platform = self.platform,
                                 chip = self.chip,
                                 sdk = self.sdk,
                                 dma_parallelization = self.dma_parallelization)
            return None
        print("  ERROR: no tiling found. Exiting...")
        os._exit(0)
        return None