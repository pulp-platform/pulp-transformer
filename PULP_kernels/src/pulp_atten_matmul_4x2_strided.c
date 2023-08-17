/*
 * pulp_atten_matmul_4x2_strided.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 *
 * Copyright (C) 2018-2021 University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pmsis.h"

#define min(a,b) ((a)<(b)?(a):(b))
#define SumDotp(a, b, c) __builtin_pulp_sdotsp4(a, b, c)
#define clip8(x) __builtin_pulp_clip_r(x, 127)

void __attribute__ ((noinline)) pulp_atten_matmul_4x2_strided(
  const int8_t * pInBuffer,
  const int8_t *  pWeight,
  int8_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  projections,
  const uint16_t  heads
) 
{
  int core_id = pi_core_id();
  int Log2Core = log2(NUM_CORES);
  int heads_per_core = (heads >> Log2Core) + ((heads & (NUM_CORES-1))!=0);
  int start_head, stop_head;
  start_head = min(heads_per_core * core_id, heads);
  stop_head = min(start_head + heads_per_core, heads);

  // local vars
  int seq_out, proj_out, seq_out_internal, head_out;
  int8_t *pA, *pA2;
  int8_t *pB, *pB2, *pB3, *pB4;
  int8_t *pOut = pOutBuffer;
  int32_t softmax_buffer_1_base[dim_sequence];
  int32_t softmax_buffer_2_base[dim_sequence];
  int32_t *softmax_buffer_1; int32_t *softmax_buffer_2;
  int8_t *pOut2 = pOut + dim_sequence * heads;
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
        *softmax_buffer_1 = sum;
        softmax_buffer_1++;
        *softmax_buffer_1 = sum2;
        softmax_buffer_1++;
        *softmax_buffer_1 = sum3;
        softmax_buffer_1++;
        *softmax_buffer_1 = sum4;
        softmax_buffer_1++;
        *softmax_buffer_2 = sum5;
        softmax_buffer_2++;
        *softmax_buffer_2 = sum6;
        softmax_buffer_2++;
        *softmax_buffer_2 = sum7;
        softmax_buffer_2++;
        *softmax_buffer_2 = sum8;
        softmax_buffer_2++;
        pB = pB + (3 * projections);
      }
      IntSoftmax(softmax_buffer_1_base, dim_sequence, pOut, pWeight);
      IntSoftmax(softmax_buffer_2_base, dim_sequence, pOut2, pWeight);
    }
  }
  pi_cl_team_barrier(0);
}
