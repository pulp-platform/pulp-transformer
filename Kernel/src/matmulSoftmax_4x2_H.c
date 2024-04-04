/* ----------------------------------------------------------------------
#
# File: matmulSoftmax_4x2_H.c
#
# Last edited: 10.10.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author:
# - Victor Jung, jungvi@iis.ee.ethz.ch, ETH Zurich
# - Alessio Burrello alessio.burrello@unibo.it, UNIBO
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

void __attribute__ ((noinline)) matmulSoftmax_4x2_H(
  const int8_t *  pInBuffer,
  const int8_t *  pWeight,
  int8_t *        pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  projections,
  const uint16_t  heads,
  const int16_t   requant_div,
  const int16_t   requant_mul,  
  const int32_t   coeffA,
  const int32_t   coeffB,
  const int32_t   coeffC,
  const int32_t   log2,
  const uint32_t  n_levels
) 
{
  int core_id = pi_core_id();
  int Log2Core = log2(NUM_CORES);

  int head_per_core = ((heads>>1) >> Log2Core) + (((heads>>1) & (NUM_CORES-1))!=0);

  int start_head, stop_head;
  start_head = min(head_per_core * core_id, heads);
  stop_head = min(start_head + head_per_core, heads);

  // local vars
  int seq_out, proj_out, seq_out_internal, head_out;
  int8_t *pA, *pA2;
  int8_t *pB, *pB2, *pB3, *pB4;
  uint8_t *pOut = pOutBuffer;
  int8_t softmax_buffer_1_base[dim_sequence];
  int8_t softmax_buffer_2_base[dim_sequence];
  int8_t *softmax_buffer_1; int8_t *softmax_buffer_2;
  uint8_t *pOut2 = pOut + dim_sequence * heads;
  int8_t seq_internal_left;
  int8_t seq_out_left;
  v4s vecA, vecA2;
  v4s vecB, vecB2, vecB3, vecB4;

  for (head_out = start_head; head_out < stop_head; head_out++)
  {  
    for (seq_out = 0; seq_out < (dim_sequence>>1); seq_out++)
    {
      pB = pWeight + (head_out * dim_sequence * projections);
      pOut = pOutBuffer + head_out * dim_sequence + seq_out * heads * dim_sequence * 2;
      pOut2 = pOut + heads * dim_sequence;
      softmax_buffer_1 = softmax_buffer_1_base;
      softmax_buffer_2 = softmax_buffer_2_base;

      for (seq_out_internal = 0; seq_out_internal < (dim_sequence>>2); seq_out_internal++)
      { 
        int sum = 0;
        int sum2 = 0;
        int sum3 = 0;
        int sum4 = 0;
        int sum5 = 0;
        int sum6 = 0;
        int sum7 = 0;
        int sum8 = 0;

        pB2 = pB + projections;
        pB3 = pB2 + projections;
        pB4 = pB3 + projections;
        
        pA = pInBuffer + head_out * dim_sequence * projections + 2 * seq_out * projections;
        pA2 = pA + projections;
        
        for (proj_out = 0; proj_out < (projections>>2); proj_out++)
        { 
          vecA = *((v4s*)pA);
          vecA2 = *((v4s*)pA2);
          vecB = *((v4s*)pB);
          vecB2 = *((v4s*)pB2);
          vecB3 = *((v4s*)pB3);
          vecB4 = *((v4s*)pB4);
        
          sum = SumDotp(vecA, vecB, sum);
          sum2 = SumDotp(vecA, vecB2, sum2);
          sum3 = SumDotp(vecA, vecB3, sum3);
          sum4 = SumDotp(vecA, vecB4, sum4);
          sum5 = SumDotp(vecA2, vecB, sum5);
          sum6 = SumDotp(vecA2, vecB2, sum6);
          sum7 = SumDotp(vecA2, vecB3, sum7);
          sum8 = SumDotp(vecA2, vecB4, sum8);
        
          pA+=4;
          pA2+=4;
          pB+=4;
          pB2+=4;
          pB3+=4;
          pB4+=4;
        }
        proj_out = projections % 4;

        while(proj_out > 0){
          sum += *pA * *pB;
          sum2 += *pA * *pB2;
          sum3 += *pA * *pB3;
          sum4 += *pA * *pB4;
          sum5 += *pA2 * *pB;
          sum6 += *pA2 * *pB2;
          sum7 += *pA2 * *pB3;
          sum8 += *pA2 * *pB4;

          pA++;
          pA2++;
          pB++;
          pB2++;
          pB3++;
          pB4++;

          proj_out--;
        }
        
        *softmax_buffer_1 = clip8((sum*requant_mul)>>requant_div);
        softmax_buffer_1++;
        *softmax_buffer_1 = clip8((sum2*requant_mul)>>requant_div);
        softmax_buffer_1++;
        *softmax_buffer_1 = clip8((sum3*requant_mul)>>requant_div);
        softmax_buffer_1++;
        *softmax_buffer_1 = clip8((sum4*requant_mul)>>requant_div);
        softmax_buffer_1++;
        *softmax_buffer_2 = clip8((sum5*requant_mul)>>requant_div);
        softmax_buffer_2++;
        *softmax_buffer_2 = clip8((sum6*requant_mul)>>requant_div);
        softmax_buffer_2++;
        *softmax_buffer_2 = clip8((sum7*requant_mul)>>requant_div);
        softmax_buffer_2++;
        *softmax_buffer_2 = clip8((sum8*requant_mul)>>requant_div);
        softmax_buffer_2++;
        
        pB = pB + (3 * projections);
      }
      seq_internal_left = dim_sequence % 4;
      
      while(seq_internal_left > 0){
        pA = pInBuffer + head_out * dim_sequence * projections + 2 * seq_out * projections;
        pA2 = pA + projections;
        
        int sum = 0;
        int sum5 = 0;
        
        for (proj_out = 0; proj_out < (projections>>2); proj_out++)
        { 
          vecA = *((v4s*)pA);
          vecA2 = *((v4s*)pA2);
          vecB = *((v4s*)pB);
      
          sum = SumDotp(vecA, vecB, sum);
          sum5 = SumDotp(vecA2, vecB, sum5);
      
          pA+=4;
          pA2+=4;
          pB+=4;
        }
        proj_out = projections % 4;

        while(proj_out > 0){
          sum += *pA * *pB;
          sum5 += *pA2 * *pB;

          pA++;
          pA2++;
          pB++;

          proj_out--;
        }
      
        *softmax_buffer_1 = clip8((sum*requant_mul)>>requant_div);
        softmax_buffer_1++;
        *softmax_buffer_2 = clip8((sum5*requant_mul)>>requant_div);
        softmax_buffer_2++;
      
        seq_internal_left -= 1;
      }
      
      iSoftmax(softmax_buffer_1_base, pOut, dim_sequence,  coeffA, coeffB, coeffC, log2, n_levels);
      iSoftmax(softmax_buffer_2_base, pOut2, dim_sequence,  coeffA, coeffB, coeffC, log2, n_levels);
    }
    seq_out_left = dim_sequence % 2;
    
    if(seq_out_left){
      pB = pWeight + (head_out * dim_sequence * projections);
      pOut = pOut2 + heads * dim_sequence;//point to the last row in the sequence
      softmax_buffer_1 = softmax_buffer_1_base;
    
      for (seq_out_internal = 0; seq_out_internal < (dim_sequence>>2); seq_out_internal++)
      {  
        int sum = 0;
        int sum2 = 0;
        int sum3 = 0;
        int sum4 = 0;
    
        pB2 = pB + projections;
        pB3 = pB2 + projections;
        pB4 = pB3 + projections;
        pA = pInBuffer + head_out * dim_sequence * projections + 2 * seq_out * projections;
    
        for (proj_out = 0; proj_out < (projections>>2); proj_out++)
        { 
          vecA = *((v4s*)pA);
          vecB = *((v4s*)pB);
          vecB2 = *((v4s*)pB2);
          vecB3 = *((v4s*)pB3);
          vecB4 = *((v4s*)pB4);
    
          sum = SumDotp(vecA, vecB, sum);
          sum2 = SumDotp(vecA, vecB2, sum2);
          sum3 = SumDotp(vecA, vecB3, sum3);
          sum4 = SumDotp(vecA, vecB4, sum4);
    
          pA+=4;
          pB+=4;
          pB2+=4;
          pB3+=4;
          pB4+=4;
        }
    
        *softmax_buffer_1 = clip8((sum*requant_mul)>>requant_div);
        softmax_buffer_1++;
        *softmax_buffer_1 = clip8((sum2*requant_mul)>>requant_div);
        softmax_buffer_1++;
        *softmax_buffer_1 = clip8((sum3*requant_mul)>>requant_div);
        softmax_buffer_1++;
        *softmax_buffer_1 = clip8((sum4*requant_mul)>>requant_div);
        softmax_buffer_1++;
    
        pB = pB + (3 * projections);
      }
    
      seq_internal_left = dim_sequence % 4;

      while(seq_internal_left > 0){
        pA = pInBuffer + head_out * dim_sequence * projections + 2 * seq_out * projections;
        int sum = 0;
    
        for (proj_out = 0; proj_out < (projections>>2); proj_out++)
        { 
          vecA = *((v4s*)pA);
          vecB = *((v4s*)pB);

          sum = SumDotp(vecA, vecB, sum);

          pA+=4;
          pB+=4;
        }
        *softmax_buffer_1 = clip8((sum*requant_mul)>>requant_div);
        softmax_buffer_1++;

        seq_internal_left -= 1;
      }
      
      iSoftmax(softmax_buffer_1_base, pOut, dim_sequence,  coeffA, coeffB, coeffC, log2, n_levels);
      seq_out_left -= 1;
    }
  }
  pi_cl_team_barrier(0);
}
