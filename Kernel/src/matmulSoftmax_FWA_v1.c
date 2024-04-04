/* ----------------------------------------------------------------------
#
# File: matmulSoftmax_FWA.c
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
*/

#include "pmsis.h"
#include "../inc/pulp_nn_utils.h"
#include "../inc/pulp_nn_kernels.h"

#define min(a,b) ((a)<(b)?(a):(b))
#define SumDotp(a, b, c) __builtin_pulp_sdotsp4(a, b, c)
#define clip8(x) __builtin_pulp_clip_r(x, 127)

void __attribute__ ((noinline)) matmulSoftmax_FWA_v1(
  const int8_t *  pInBuffer,
  const int8_t *  pWeight,
  const int16_t * pBias,
  int8_t *        pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  dim_embedding,
  const uint16_t  heads,
  const int16_t   pre_proj_requant_div,
  const int16_t   pre_proj_requant_mul,
  const int16_t   post_proj_requant_div,
  const int16_t   post_proj_requant_mul,  
  const int32_t   coeffA,
  const int32_t   coeffB,
  const int32_t   coeffC,
  const int32_t   log2,
  const uint32_t  n_levels
){

  int start_head, stop_head;
  int accumulator;
  int8_t *pInOriginal = pInBuffer;
  int8_t *pWeightOriginal = pWeight;
  int16_t *pBiasOriginal = pBias;
  int8_t *pOutBufferOriginal = pOutBuffer;

  int8_t *intermediateBuffer = pOutBufferOriginal + 100000;
  int8_t *intermediateBufferOriginal = intermediateBuffer;

  int8_t softmax_buffer[dim_sequence];
  int8_t *softmax_buffer_ptr = softmax_buffer;

  int head_per_core = (heads >> log2(NUM_CORES)) + ((heads & (NUM_CORES-1)) != 0);
  start_head = min(head_per_core * pi_core_id(), heads);
  stop_head = min(start_head + head_per_core, heads);
    
  for (int h = start_head; h < stop_head; h++){
    intermediateBuffer = intermediateBufferOriginal + h*dim_sequence*dim_embedding;
    for (int s = 0; s < dim_sequence; s++){
      pWeight = pWeightOriginal + h*dim_embedding*dim_embedding;
      pBias = pBiasOriginal + h*dim_embedding;
      for (int e0 = 0; e0 < dim_embedding; e0++){
        pInBuffer = pInOriginal + s*dim_embedding;
        accumulator = *pBias;
        pBias++;
        for (int e1 = 0; e1 < dim_embedding; e1++){
          accumulator += (*pInBuffer) * (*pWeight);
          pInBuffer++;
          pWeight++;
        }
        *intermediateBuffer = clip8((accumulator*pre_proj_requant_mul)>>pre_proj_requant_div);
        intermediateBuffer++;
      }
    }
  }
  pi_cl_team_barrier(0); 

  for (int h = start_head; h < stop_head; h++){
    pOutBuffer = pOutBufferOriginal + h*dim_sequence*dim_sequence;
    for (int s0 = 0; s0 < dim_sequence; s0++){
      pInBuffer = pInOriginal;
      for (int s1 = 0; s1 < dim_sequence; s1++){
        intermediateBuffer = intermediateBufferOriginal + h*dim_sequence*dim_embedding + s0*dim_embedding;
        accumulator = 0;
        for (int e = 0; e < dim_embedding; e++){
          accumulator += (*intermediateBuffer) * (*pInBuffer);
          intermediateBuffer++;
          pInBuffer++;
        }
        *softmax_buffer_ptr = clip8((accumulator*post_proj_requant_mul)>>post_proj_requant_div);
        softmax_buffer_ptr++;
      }
      iSoftmax(softmax_buffer, pOutBuffer, dim_sequence,  coeffA, coeffB, coeffC, log2, n_levels);
      softmax_buffer_ptr = softmax_buffer_ptr - dim_sequence;
      pOutBuffer += dim_sequence;
    }
  }
}

