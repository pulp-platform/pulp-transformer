/* ----------------------------------------------------------------------
#
# File: matmul_4x2_H.c
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
// #define SumDotp(a, b, c) __builtin_pulp_sdotsp4(a, b, c)
#define SumDotp(a, b, c) __builtin_pulp_sdotusp4(a, b, c)
#define clip8(x) __builtin_pulp_clip_r(x, 127)

void __attribute__ ((noinline)) matmul_4x2_H(
  const int8_t * pInBuffer,
  const int8_t *  pWeight,
  int8_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  projections,
  const uint16_t  heads,
  const int16_t   requant_div,
  const int16_t   requant_mul
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
  uint8_t *pA, *pA2;
  int8_t *pB, *pB2, *pB3, *pB4;
  int8_t *pOut = pOutBuffer;
  int8_t *pOut2 = pOut + dim_sequence * heads;
  int32_t offset = 128;
  int32_t sumB_offset, sumB2_offset, sumB3_offset, sumB4_offset;
  v4u vecA, vecA2;
  v4s vecB, vecB2, vecB3, vecB4;

  for (head_out = start_head; head_out < stop_head; head_out++)
  {  
    for (seq_out = 0; seq_out < (dim_sequence>>1); seq_out++)
    {
      pB = pWeight + (head_out * dim_sequence * projections);
      pOut = pOutBuffer + head_out * projections + seq_out * heads * projections * 2;
      pOut2 = pOut + heads * projections;

      for (proj_out = 0; proj_out < (projections>>2); proj_out++)
      {  
        int sum = 0;
        int sum2 = 0;
        int sum3 = 0;
        int sum4 = 0;
        int sum5 = 0;
        int sum6 = 0;
        int sum7 = 0;
        int sum8 = 0;
      
        pB2 = pB + dim_sequence;
        pB3 = pB2 + dim_sequence;
        pB4 = pB3 + dim_sequence;
        pA = pInBuffer + head_out * dim_sequence + 2 * seq_out * heads * dim_sequence;
        pA2 = pA + heads * dim_sequence;
      
        for (seq_out_internal = 0; seq_out_internal < (dim_sequence>>2); seq_out_internal++)
        { 
          vecA = *((v4u*)pA);
          vecA2 = *((v4u*)pA2);
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

        int seq_internal_left = dim_sequence % 4;
        while(seq_internal_left>0){

          sum = sum + ((*pA)) * (*pB);
          sum2 = sum2 + ((*pA)) * (*pB2);
          sum3 = sum3 + ((*pA)) * (*pB3);
          sum4 = sum4 + ((*pA++)) * (*pB4);
          sum5 = sum5 + ((*pA2)) * (*pB++);
          sum6 = sum6 + ((*pA2)) * (*pB2++);
          sum7 = sum7 + ((*pA2)) * (*pB3++);
          sum8 = sum8 + ((*pA2++)) * (*pB4++);

          seq_internal_left -= 1;
        }
        
        *pOut = clip8((sum*requant_mul)>>requant_div);
        pOut++;
        *pOut = clip8((sum2*requant_mul)>>requant_div);
        pOut++;
        *pOut = clip8((sum3*requant_mul)>>requant_div);
        pOut++;
        *pOut = clip8((sum4*requant_mul)>>requant_div);
        pOut++;
        *pOut2 = clip8((sum5*requant_mul)>>requant_div);
        pOut2++;
        *pOut2 = clip8((sum6*requant_mul)>>requant_div);
        pOut2++;
        *pOut2 = clip8((sum7*requant_mul)>>requant_div);
        pOut2++;
        *pOut2 = clip8((sum8*requant_mul)>>requant_div);
        pOut2++;
        pB = pB + (3 * dim_sequence);
      }
      proj_out = projections % 4;
      
      while(proj_out > 0){
        int sum = 0;
        int sum5 = 0;

        pA = pInBuffer + head_out * dim_sequence + 2 * seq_out * heads * dim_sequence;
        pA2 = pA + heads * dim_sequence;

        for (seq_out_internal = 0; seq_out_internal < (dim_sequence>>2); seq_out_internal++)
        { 
          vecA = *((v4u*)pA);
          vecA2 = *((v4u*)pA2);
          vecB = *((v4s*)pB);

          sum = SumDotp(vecA, vecB, sum);
          sum5 = SumDotp(vecA2, vecB, sum5);

          pA+=4;
          pA2+=4;
          pB+=4;
        }

        int seq_internal_left = dim_sequence % 4;
        while(seq_internal_left>0){
          sum = sum + ((*pA++)) * (*pB);
          sum5 = sum5 + ((*pA2++)) * (*pB++);

          seq_internal_left -= 1;
        }

        *pOut = clip8((sum*requant_mul)>>requant_div);
        pOut++;
        *pOut2 = clip8((sum5*requant_mul)>>requant_div);
        pOut2++;
      
        proj_out -= 1;
      }
    }

    int seq_out_left = dim_sequence % 2;  
    if (seq_out_left){  
      pB = pWeight + (head_out * dim_sequence * projections);
      pOut = pOut2;
      
      for (proj_out = 0; proj_out < (projections>>2); proj_out++)
      {  
        int sum = 0;
        int sum2 = 0;
        int sum3 = 0;
        int sum4 = 0;
        int sum5 = 0;
        int sum6 = 0;
        int sum7 = 0;
        int sum8 = 0;
      
        pB2 = pB + dim_sequence;
        pB3 = pB2 + dim_sequence;
        pB4 = pB3 + dim_sequence;
        pA = pInBuffer + head_out * dim_sequence + 2 * seq_out * heads * dim_sequence;
      
        for (seq_out_internal = 0; seq_out_internal < (dim_sequence>>2); seq_out_internal++)
        { 
          vecA = *((v4u*)pA);
          vecA2 = *((v4u*)pA2);
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
        int seq_internal_left = dim_sequence % 4;
      
        while(seq_internal_left>0){
          sum = sum + ((*pA)) * (*pB++);
          sum2 = sum2 + ((*pA)) * (*pB2++);
          sum3 = sum3 + ((*pA)) * (*pB3++);
          sum4 = sum4 + ((*pA++)) * (*pB4++);

          seq_internal_left -= 1;
        }
        *pOut = clip8((sum*requant_mul)>>requant_div);
        pOut++;
        *pOut = clip8((sum2*requant_mul)>>requant_div);
        pOut++;
        *pOut = clip8((sum3*requant_mul)>>requant_div);
        pOut++;
        *pOut = clip8((sum4*requant_mul)>>requant_div);
        pOut++;
     
        pB = pB + (3 * dim_sequence);
      }

      proj_out = projections % 4;
      
      while(proj_out > 0){
        int sum = 0;

        pA = pInBuffer + head_out * dim_sequence + 2 * seq_out * heads * dim_sequence;

        for (seq_out_internal = 0; seq_out_internal < (dim_sequence>>2); seq_out_internal++)
        { 
          vecA = *((v4u*)pA);
          vecB = *((v4s*)pB);

          sum = SumDotp(vecA, vecB, sum);

          pA+=4;
          pB+=4;
        }

        int seq_internal_left = dim_sequence % 4;

        while(seq_internal_left>0){
          sum = sum + ((*pA++)) * (*pB++);
          seq_internal_left -= 1;
        }

        *pOut = clip8((sum*requant_mul)>>requant_div);
        pOut++;
      
        proj_out -= 1;
      }
     seq_out_left -= 1;
    }
  }
}
