/* ----------------------------------------------------------------------
#
# File: matmulSoftmax_FWA_v3_H.c
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
*/


#include "pmsis.h"
#include "../inc/pulp_nn_utils.h"
#include "../inc/pulp_nn_kernels.h"

#define min(a,b) ((a)<(b)?(a):(b))
#define SumDotp(a, b, c) __builtin_pulp_sdotsp4(a, b, c)
#define clip8(x) __builtin_pulp_clip_r(x, 127)

void __attribute__ ((noinline)) matmulSoftmax_FWA_v3_H(
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
  int acc1, acc2, acc3, acc4; // Accumulators
  v4s vecI1, vecI2, vecW1, vecW2;

  // Keep in mind original pointers
  int8_t *pInOriginal = pInBuffer;
  int8_t *pWeightOriginal = pWeight;
  int16_t *pBiasOriginal = pBias;
  int8_t *pOutBufferOriginal = pOutBuffer;

  int8_t *pIn1, *pIn2;
  int8_t *pW1, *pW2;
  int8_t *pOut1, *pOut2;

  int8_t *intermediateBuffer = pOutBufferOriginal + 100000;
  int8_t *intermediateBufferOriginal = intermediateBuffer;

  int8_t *pInter1, *pInter2;

  int8_t softmax_buffer1[dim_sequence];
  int8_t *softmax_buffer1_ptr = softmax_buffer1;
  int8_t softmax_buffer2[dim_sequence];
  int8_t *softmax_buffer2_ptr = softmax_buffer2;

  int head_per_core = (heads >> log2(NUM_CORES)) + ((heads & (NUM_CORES-1)) != 0);
  start_head = min(head_per_core * pi_core_id(), heads);
  stop_head = min(start_head + head_per_core, heads);
    
  for (int h = start_head; h < stop_head; h++){
    pInter1 = intermediateBufferOriginal + h*dim_sequence*dim_embedding;
    pInter2 = pInter1 + dim_embedding;

    for (int s = 0; s < (dim_sequence >> 1); s++){
      pW1 = pWeightOriginal + h*dim_embedding*dim_embedding;
      pW2 = pW1 + dim_embedding;
      pBias = pBiasOriginal + h*dim_embedding;

      for (int e0 = 0; e0 < (dim_embedding >> 1); e0++){
        pIn1 = pInOriginal + 2*s*dim_embedding;
        pIn2 = pIn1 + dim_embedding;
        acc1 = *pBias;
        acc2 = *pBias;
        pBias++;
        acc3 = *pBias;
        acc4 = *pBias;
        pBias++;

        for (int e1 = 0; e1 < (dim_embedding >> 2); e1++){
          vecI1 = *((v4s*) pIn1);
          vecI2 = *((v4s*) pIn2);
          vecW1 = *((v4s*) pW1);
          vecW2 = *((v4s*) pW2);

          acc1 = SumDotp(vecI1, vecW1, acc1);
          acc2 = SumDotp(vecI2, vecW1, acc2);
          acc3 = SumDotp(vecI1, vecW2, acc3);
          acc4 = SumDotp(vecI2, vecW2, acc4);

          pIn1 += 4;
          pIn2 += 4;
          pW1 += 4;
          pW2 += 4;
        }
        *pInter1 = clip8((acc1*pre_proj_requant_mul)>>pre_proj_requant_div);
        pInter1++;
        *pInter1 = clip8((acc3*pre_proj_requant_mul)>>pre_proj_requant_div);
        pInter1++;

        *pInter2 = clip8((acc2*pre_proj_requant_mul)>>pre_proj_requant_div);
        pInter2++; 
        *pInter2 = clip8((acc4*pre_proj_requant_mul)>>pre_proj_requant_div);
        pInter2++;

        pW1 += dim_embedding;
        pW2 += dim_embedding;
      }

      pInter1 += dim_embedding;
      pInter2 += dim_embedding;
    }
  }
  pi_cl_team_barrier(0); 

  for (int h = start_head; h < stop_head; h++){
    pOut1 = pOutBufferOriginal + h*dim_sequence*dim_sequence;
    pOut2 = pOut1 + dim_sequence;

    for (int s0 = 0; s0 < (dim_sequence >> 1); s0++){
      pIn1 = pInOriginal;
      pIn2 = pIn1 + dim_embedding;

      for (int s1 = 0; s1 < (dim_sequence >> 1); s1++){
        pInter1 = intermediateBufferOriginal + h*dim_sequence*dim_embedding + 2*s0*dim_embedding;
        pInter2 = pInter1 + dim_embedding;
        acc1 = 0;
        acc2 = 0;
        acc3 = 0;
        acc4 = 0;

        for (int e = 0; e < (dim_embedding >> 2); e++){
          vecI1 = *((v4s*) pIn1);
          vecI2 = *((v4s*) pIn2);
          vecW1 = *((v4s*) pInter1);
          vecW2 = *((v4s*) pInter2);

          acc1 = SumDotp(vecW1, vecI1, acc1);
          acc2 = SumDotp(vecW2, vecI1, acc2);
          acc3 = SumDotp(vecW1, vecI2, acc3);
          acc4 = SumDotp(vecW2, vecI2, acc4);
          
          pInter1 += 4;
          pInter2 += 4;
          pIn1 += 4;
          pIn2 += 4;
        }
        *softmax_buffer1_ptr = clip8((acc1*post_proj_requant_mul)>>post_proj_requant_div);
        softmax_buffer1_ptr++;
        *softmax_buffer1_ptr = clip8((acc3*post_proj_requant_mul)>>post_proj_requant_div);
        softmax_buffer1_ptr++;

        *softmax_buffer2_ptr = clip8((acc2*post_proj_requant_mul)>>post_proj_requant_div);
        softmax_buffer2_ptr++;
        *softmax_buffer2_ptr = clip8((acc4*post_proj_requant_mul)>>post_proj_requant_div);
        softmax_buffer2_ptr++;

        pIn1 += dim_embedding;
        pIn2 += dim_embedding;
      }

      iSoftmax(softmax_buffer1, pOut1, dim_sequence,  coeffA, coeffB, coeffC, log2, n_levels);
      iSoftmax(softmax_buffer2, pOut2, dim_sequence,  coeffA, coeffB, coeffC, log2, n_levels);

      softmax_buffer1_ptr -= dim_sequence;
      softmax_buffer2_ptr -= dim_sequence;

      pOut1 += 2*dim_sequence;
      pOut2 = pOut1 + dim_sequence;
    }
  }
}

