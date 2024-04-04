/*
 * pulp_atten_linear_2x1_input_reuse.c
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

void __attribute__ ((noinline)) pulp_atten_linear_2x1_input_reuse(
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
  int8_t *pA;
  int8_t *pB, *pB2;
  int8_t *pOut = pOutBuffer + start_head * projections * dim_sequence;
  v4s vecA;
  v4s vecB;
  v4s vecB2;

  for (head_out = start_head; head_out < stop_head; head_out++)
  {
    for (seq_out = 0; seq_out < dim_sequence; seq_out++)
    {  
      pB = pWeight + (head_out * dim_embedding * projections);
      for (proj_out = 0; proj_out < (projections>>1); proj_out++)
      {  
        int sum = 0;
        int sum2 = 0;
        pB2 = pB + dim_embedding;
        pA = pInBuffer + seq_out * dim_embedding;
        for (emb = 0; emb < (dim_embedding>>2); emb++)
        { 
          vecA = *((v4s*)pA);
          vecB = *((v4s*)pB);
          vecB2 = *((v4s*)pB2);
          sum = SumDotp(vecA, vecB, sum);
          sum2 = SumDotp(vecA, vecB2, sum2);
          pA+=4;
          pB+=4;
          pB2+=4;
        }
        *pOut = clip8(sum);
        pOut++;
        *pOut = clip8(sum2);
        pOut++;
        pB = pB + (dim_embedding);
      }
    }
  }
  pi_cl_team_barrier(0);
}
