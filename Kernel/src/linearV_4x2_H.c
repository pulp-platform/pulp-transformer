/* ----------------------------------------------------------------------
#
# File: linearV_4x2_H.c
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

void __attribute__ ((noinline)) linearV_4x2_H(
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
  // printf("Weights Qlinear:\n");
  // for(int i = 0; i < 10; i++){
  //   printf("%d \t", pWeight[i]);
  // }
  // printf("\n");
  // for(int i = 256-10; i < 256; i++){
  //   printf("%d \t", pBias[i]);
  // }
  // printf("\n");
  // printf("Requant div and mul: %d %d\n", requant_div, requant_mul);

  // We spatially unroll the heads over the GAP8 cores
  int8_t core_id = pi_core_id();
  int8_t Log2Core = log2(NUM_CORES);
  int8_t heads_per_core = (heads >> Log2Core) + ((heads & (NUM_CORES-1))!=0);
  int8_t start_head, stop_head;
  start_head = min(heads_per_core * core_id, heads);
  stop_head = min(start_head + heads_per_core, heads);

  // local vars
  int32_t head_out, proj_out, seq_out, emb;
  int32_t seq_leftover;
  int8_t *pA, *pA2, *pA3, *pA4;
  int8_t *pB, *pB2;
  int8_t *pOut = pOutBuffer + start_head * projections * dim_sequence;
  int8_t *pOut2 = pOut + dim_sequence;
  v4s vecA, vecA2, vecA3, vecA4;
  v4s vecB, vecB2;
  int32_t sum, sum2, sum3, sum4, sum5, sum6, sum7, sum8; // Accumulators
  int16_t *pBias;

  // for(int i = 0; i < 256; i++){
  //   printf("%d \t", pBiasBuffer[i]);
  // }

  // We spatially unroll the 2 sequences and 4 projections within one GAP8 core
  for (head_out = start_head; head_out < stop_head; head_out++)
  {
    for (proj_out = 0; proj_out < (projections>>1); proj_out++)
    {  
      pA = pInBuffer;
      pOut = pOutBuffer + (head_out * projections * dim_sequence) + (2 * proj_out * dim_sequence);
      pOut2 = pOut + dim_sequence;

      for (seq_out = 0; seq_out < (dim_sequence>>2); seq_out++)
      { 

        pBias = pBiasBuffer + (head_out * projections) + proj_out*2; 
        sum = *pBias;
        sum2 = sum;
        sum3 = sum;
        sum4 = sum;
        pBias++;

        sum5 = *pBias;
        sum6 = sum5;
        sum7 = sum5;
        sum8 = sum5;

        pA2 = pA + dim_embedding;
        pA3 = pA2 + dim_embedding;
        pA4 = pA3 + dim_embedding;
        pB = pWeight + (head_out * dim_embedding * projections) + (dim_embedding * proj_out * 2);
        pB2 = pB + dim_embedding;
        for (emb = 0; emb < (dim_embedding>>2); emb++)
        { 
          vecA = *((v4s*)pA);
          vecA2 = *((v4s*)pA2);
          vecA3 = *((v4s*)pA3);
          vecA4 = *((v4s*)pA4);
          vecB = *((v4s*)pB);
          vecB2 = *((v4s*)pB2);
          sum = SumDotp(vecA, vecB, sum);
          sum2 = SumDotp(vecA2, vecB, sum2);
          sum3 = SumDotp(vecA3, vecB, sum3);
          sum4 = SumDotp(vecA4, vecB, sum4);
          sum5 = SumDotp(vecA, vecB2, sum5);
          sum6 = SumDotp(vecA2, vecB2, sum6);
          sum7 = SumDotp(vecA3, vecB2, sum7);
          sum8 = SumDotp(vecA4, vecB2, sum8);
          pA+=4;
          pA2+=4;
          pA3+=4;
          pA4+=4;
          pB+=4;
          pB2+=4;
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
        pA = pA + (3 * dim_embedding);
      }
      
      // Compute remaining sequence temporally
      seq_out = dim_sequence % 4;

      while(seq_out > 0){
        
        pBias = pBiasBuffer + (head_out * projections) + proj_out*2; 
        pB = pWeight + (head_out * dim_embedding * projections) + (dim_embedding * proj_out * 2);
        pB2 = pB + dim_embedding;
        
        sum = *pBias;
        pBias++;
        sum5 = *pBias;
        
        for (emb = 0; emb < (dim_embedding>>2); emb++)
        {

          vecA = *((v4s*)pA);
          vecB = *((v4s*)pB);
          vecB2 = *((v4s*)pB2);

          sum = SumDotp(vecA, vecB, sum);
          sum5 = SumDotp(vecA, vecB2, sum5);

          pA+=4;
          pB+=4;
          pB2+=4;
        }

        *pOut = clip8((sum*requant_mul)>>requant_div);
        pOut++;
        *pOut2 = clip8((sum5*requant_mul)>>requant_div);
        pOut2++;
        
        seq_out -= 1;
      }
    }

    // Compute remaining projections temporally
    proj_out = projections % 2;
    
    if(proj_out){
      
      pBias = pBiasBuffer + (head_out * projections) + projections - proj_out; //point to last bias

      for (seq_out = 0; seq_out < (dim_sequence>>2); seq_out++)
      {

        sum = *pBias;
        sum2 = sum;
        sum3 = sum;
        sum4 = sum;

        pA2 = pA + dim_embedding;
        pA3 = pA2 + dim_embedding;
        pA4 = pA3 + dim_embedding;
        pB = pWeight + (head_out * dim_embedding * projections) + ((projections - proj_out) * dim_embedding); //last projection


        for (emb = 0; emb < (dim_embedding>>2); emb++)
        {
          vecA = *((v4s*)pA);
          vecA2 = *((v4s*)pA2);
          vecA3 = *((v4s*)pA3);
          vecA4 = *((v4s*)pA4);
          vecB = *((v4s*)pB);
          sum = SumDotp(vecA, vecB, sum);
          sum2 = SumDotp(vecA2, vecB, sum2);
          sum3 = SumDotp(vecA3, vecB, sum3);
          sum4 = SumDotp(vecA4, vecB, sum4);

          pA+=4;
          pA2+=4;
          pA3+=4;
          pA4+=4;
          pB+=4;
        }

        *pOut2 = clip8((sum*requant_mul)>>requant_div);
        pOut2++;
        *pOut2 = clip8((sum2*requant_mul)>>requant_div);
        pOut2++;
        *pOut2 = clip8((sum3*requant_mul)>>requant_div);
        pOut2++;
        *pOut2 = clip8((sum4*requant_mul)>>requant_div);
        pOut2++;

      }
    }
  }
  // for(int i=0; i<81*256; i++) {
  //   printf("%d ", pOutBuffer[i]);
  // }
  // printf("\n");
}
