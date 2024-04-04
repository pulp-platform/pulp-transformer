/* ----------------------------------------------------------------------
#
# File: linearQK_4x2_H.c
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
#include "math.h"

#define min(a,b) ((a)<(b)?(a):(b))
#define SumDotp(a, b, c) __builtin_pulp_sdotsp4(a, b, c)
#define clip8(x) __builtin_pulp_clip_r(x, 127)

void __attribute__ ((noinline)) linearQK_4x2_H(
  const int8_t *  pInBuffer,
  const int8_t *  pWeight,
  const int16_t * pBiasBuffer,
  int8_t *        pOutBuffer,
  const uint16_t  dimSequence,
  const uint16_t  dimEmbedding,
  const uint16_t  dimProjections,
  const uint16_t  heads,
  const int16_t   requant_div,
  const int16_t   requant_mul
) 
{

  // We spatially unroll the heads over the GAP8 cores
  int8_t core_id = pi_core_id();
  int8_t Log2Core = log2(NUM_CORES);
  int8_t heads_per_core = (heads >> Log2Core) + ((heads & (NUM_CORES-1))!=0);
  int8_t start_head, stop_head;
  start_head = min(heads_per_core * core_id, heads);
  stop_head = min(start_head + heads_per_core, heads);

  // Local variables declarations
  int32_t head_out, proj_out, seq_out, emb;
  int32_t seq_leftover;
  int8_t *pA, *pA2;
  int8_t *pB, *pB2, *pB3, *pB4;
  v4s vecA, vecA2;
  v4s vecB, vecB2, vecB3, vecB4;
  int8_t *pOut, *pOut2;
  int32_t sum, sum2, sum3, sum4, sum5, sum6, sum7, sum8; // Accumulators
  int16_t *pBias;

  // We spatially unroll the 2 sequences and 4 projections within one GAP8 core
  for (head_out = start_head; head_out < stop_head; head_out++)
  {
    for (seq_out = 0; seq_out < (dimSequence>>1); seq_out++)
    {  
      
      pOut = pOutBuffer + (head_out * dimProjections * dimSequence) + (2 * seq_out * dimProjections);
      pOut2 = pOut + dimProjections;

      pBias = pBiasBuffer + (head_out * dimProjections);

      for (proj_out = 0; proj_out < (dimProjections>>2); proj_out++)
      {  
        sum = *pBias;
        pBias++;
        sum2 = *pBias;
        pBias++;
        sum3 = *pBias;
        pBias++;
        sum4 = *pBias;
        pBias++;

        sum5 = sum;
        sum6 = sum2;
        sum7 = sum3;
        sum8 = sum4;

        pA = pInBuffer + (2 * seq_out * dimEmbedding);
        pA2 = pA + dimEmbedding;

        pB = pWeight + (head_out * dimEmbedding * dimProjections) + (4 * proj_out * dimEmbedding);
        pB2 = pB + dimEmbedding;
        pB3 = pB2 + dimEmbedding;
        pB4 = pB3 + dimEmbedding;

        for (emb = 0; emb < (dimEmbedding>>2); emb++)
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
      }
      // Compute remaining projections temporaly
      proj_out = dimProjections % 4;

      while(proj_out > 0){
        
        sum = *pBias;
        sum2 = sum;
        pBias++;

        pA = pInBuffer + (2 * seq_out * dimEmbedding);
        pA2 = pA + dimEmbedding;

        pB = pWeight + (head_out * dimEmbedding * dimProjections) + ((dimProjections - proj_out) * dimEmbedding);

        for (emb = 0; emb < (dimEmbedding>>2); emb++)
        {

          vecA = *((v4s*)pA);
          vecA2 = *((v4s*)pA2);
          vecB = *((v4s*)pB);

          sum = SumDotp(vecA, vecB, sum);
          sum2 = SumDotp(vecA2, vecB, sum2);

          pA+=4;
          pA2+=4;

          pB+=4;

        }
        *pOut = clip8((sum*requant_mul)>>requant_div);
        pOut++;

        *pOut2 = clip8((sum2*requant_mul)>>requant_div);
        pOut2++;

        proj_out -= 1;
      }
    }

    // Compute remaining sequences temporaly
    seq_out = dimSequence % 2;
    
    if(seq_out){
      
      pBias = pBiasBuffer + (head_out * dimProjections);

      for (proj_out = 0; proj_out < (dimProjections>>2); proj_out++)
      {

        sum5 = *pBias;
        pBias++;
        sum6 = *pBias;
        pBias++;
        sum7 = *pBias;
        pBias++;
        sum8 = *pBias;
        pBias++;

        pA2 = pA + dimEmbedding;

        pB = pWeight + (head_out * dimEmbedding * dimProjections) + (4 * proj_out * dimEmbedding);
        pB2 = pB + dimEmbedding;
        pB3 = pB2 + dimEmbedding;
        pB4 = pB3 + dimEmbedding;

        for (emb = 0; emb < (dimEmbedding>>2); emb++)
        {
          vecA2 = *((v4s*)pA2);
          vecB = *((v4s*)pB);
          vecB2 = *((v4s*)pB2);
          vecB3 = *((v4s*)pB3);
          vecB4 = *((v4s*)pB4);

          sum5 = SumDotp(vecA2, vecB, sum5);
          sum6 = SumDotp(vecA2, vecB2, sum6);
          sum7 = SumDotp(vecA2, vecB3, sum7);
          sum8 = SumDotp(vecA2, vecB4, sum8);

          pA2+=4;

          pB+=4;
          pB2+=4;
          pB3+=4;
          pB4+=4;
        }

        *pOut2 = clip8((sum5*requant_mul)>>requant_div);
        pOut2++;
        *pOut2 = clip8((sum6*requant_mul)>>requant_div);
        pOut2++;
        *pOut2 = clip8((sum7*requant_mul)>>requant_div);
        pOut2++;
        *pOut2 = clip8((sum8*requant_mul)>>requant_div);
        pOut2++;

      }
    }
  }
  pi_cl_team_barrier(0);

}
