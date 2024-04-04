/* ----------------------------------------------------------------------
#
# File: linearO_4x2_H.c
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

void __attribute__ ((noinline)) linearO_4x2_H(
  const int8_t * pInBuffer,
  const int8_t *  pWeight,
  const int16_t *  pBiasBuffer,
  int8_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  dim_embedding,
  const uint16_t  projections,
  const uint16_t  heads,
  const int16_t   requant_div,
  const int16_t   requant_mul
) 
{
  int core_id = pi_core_id();
  int Log2Core = log2(NUM_CORES);

  int seq_per_core = ((dim_sequence>>1) >> Log2Core) + (((dim_sequence>>1) & (NUM_CORES-1))!=0);
  int leftover_seq = (dim_sequence % seq_per_core) * (core_id == (NUM_CORES-1));

  int start_seq, stop_seq;
  start_seq = min(seq_per_core * core_id, dim_sequence);
  stop_seq = min(start_seq + seq_per_core, dim_sequence);

  // local vars
  int proj_head_in, seq_out, emb_out;
  int8_t *pA, *pA2;
  int8_t *pB, *pB2, *pB3, *pB4;
  int8_t *pOut = pOutBuffer;
  int8_t *pOut2 = pOut + dim_embedding;
  int16_t *pBias;
  v4s vecA, vecA2;
  v4s vecB, vecB2, vecB3, vecB4;

  for (seq_out = start_seq; seq_out < stop_seq; seq_out++)
  {  
    pOut = pOutBuffer + seq_out * dim_embedding * 2;
    pOut2 = pOut + dim_embedding;
    pB = pWeight;
    pBias = pBiasBuffer;

    // TODO: update pout depending on start and stop emb! 
    for (emb_out = 0; emb_out < (dim_embedding>>2); emb_out++)
    {
      int sum = *pBias;
      pBias++;
      int sum2 = *pBias;
      pBias++;
      int sum3 = *pBias;
      pBias++;
      int sum4 = *pBias;
      pBias++;
      int sum5 = sum;
      int sum6 = sum2;
      int sum7 = sum3;
      int sum8 = sum4;

      pB2 = pB + heads * projections;
      pB3 = pB2 + heads * projections;
      pB4 = pB3 + heads * projections;
      pA = pInBuffer + (2 * seq_out * projections * heads);
      pA2 = pA + projections * heads;
      for (proj_head_in = 0; proj_head_in < (projections*heads)>>2; proj_head_in++)
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
      pB = pB + (3 * heads * projections);
    }
  }
  int seq_left = leftover_seq;
  if (seq_left){
    pOut = pOut2;
    pB = pWeight;
    pBias = pBiasBuffer;

    for (emb_out = 0; emb_out < (dim_embedding>>2); emb_out++)
    {
      int sum = *pBias;
      pBias++;
      int sum2 = *pBias;
      pBias++;
      int sum3 = *pBias;
      pBias++;
      int sum4 = *pBias;
      pBias++;

      pB2 = pB + heads * projections;
      pB3 = pB2 + heads * projections;
      pB4 = pB3 + heads * projections;
      pA = pA2;
      for (proj_head_in = 0; proj_head_in < (projections*heads)>>2; proj_head_in++)
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
      *pOut = clip8((sum*requant_mul)>>requant_div);
      pOut++;
      *pOut = clip8((sum2*requant_mul)>>requant_div);
      pOut++;
      *pOut = clip8((sum3*requant_mul)>>requant_div);
      pOut++;
      *pOut = clip8((sum4*requant_mul)>>requant_div);
      pOut++;
      pB = pB + (3 * heads * projections);
    }
  seq_left -= 1;
  }
  pi_cl_team_barrier(0);

  // if(pi_core_id()==0){
  //   for(int i = 0; i < 1*81*32; i++){
  //       printf("%d ", (int8_t)pOutBuffer[i]);
  //   }
  // }

}
