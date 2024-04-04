/*
 * pulp_atten_matmul_2x1_strided.c
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

void __attribute__ ((noinline)) pulp_atten_matmul_2x1_strided(
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
  int sequences_per_core = (dim_sequence >> Log2Core) + ((dim_sequence & (NUM_CORES-1))!=0);
  int start_seq, stop_seq;
  start_seq = min(sequences_per_core * core_id, dim_sequence);
  stop_seq = min(start_seq + sequences_per_core, dim_sequence);

  // local vars
  int seq_out, proj_out, seq_out_internal, head_out;
  int8_t *pA;
  int8_t *pB, *pB2;
  int8_t *pOut = pOutBuffer + start_seq * heads * dim_sequence;
  v4s vecA;
  v4s vecB;
  v4s vecB2;

  for (seq_out = start_seq; seq_out < stop_seq; seq_out++)
  {
    for (head_out = 0; head_out < heads; head_out++)
    {  
      pB = pWeight + (head_out * dim_sequence * projections);
      for (seq_out_internal = 0; seq_out_internal < (dim_sequence>>1); seq_out_internal++)
      {  
        int sum = 0;
        int sum2 = 0;
        pB2 = pB + projections;
        pA = pInBuffer + head_out * dim_sequence * projections + seq_out * projections;
        for (proj_out = 0; proj_out < (projections>>2); proj_out++)
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
        pB = pB + (projections);
      }
    }
  }
  pi_cl_team_barrier(0);
}
