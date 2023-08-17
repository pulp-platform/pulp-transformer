/*
 * pulp_atten_linear_4x2_weights_reuse.c
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

void __attribute__ ((noinline)) pulp_atten_linear_4x2_weights_reuse(
  const int8_t * pInBuffer,
  const int8_t *  pWeight,
  int8_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  dim_embedding,
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
  int head_out, proj_out, seq_out, emb;
  int8_t *pA, *pA2, *pA3, *pA4;
  int8_t *pB, *pB2;
  int8_t *pOut = pOutBuffer + start_head * projections * dim_sequence;
  int8_t *pOut2 = pOut + dim_sequence;
  v4s vecA, vecA2, vecA3, vecA4;
  v4s vecB, vecB2;

  for (head_out = start_head; head_out < stop_head; head_out++)
  {
    for (proj_out = 0; proj_out < (projections>>1); proj_out++)
    {  
      pA = pInBuffer;
      pOut = pOutBuffer + head_out * projections * dim_sequence + 2 * proj_out * dim_sequence;
      pOut2 = pOut + dim_sequence;
      for (seq_out = 0; seq_out < (dim_sequence>>2); seq_out++)
      {  
        int sum = 0;
        int sum2 = 0;
        int sum3 = 0;
        int sum4 = 0;
        int sum5 = 0;
        int sum6 = 0;
        int sum7 = 0;
        int sum8 = 0;
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
        *pOut = clip8(sum);
        pOut++;
        *pOut = clip8(sum2);
        pOut++;
        *pOut = clip8(sum3);
        pOut++;
        *pOut = clip8(sum4);
        pOut++;
        *pOut2 = clip8(sum5);
        pOut2++;
        *pOut2 = clip8(sum6);
        pOut2++;
        *pOut2 = clip8(sum7);
        pOut2++;
        *pOut2 = clip8(sum8);
        pOut2++;
        pA = pA + (3 * dim_embedding);
      }
    }
  }
  pi_cl_team_barrier(0);
}
