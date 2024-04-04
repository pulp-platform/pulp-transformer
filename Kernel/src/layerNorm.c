/*
 * pulp_nn_add.c
 * Nazareno Bruschi <nazareno.bruschi@unibo.it>
 * Angelo Garofalo <angelo.garofalo@unibo.it>
 *
 * Copyright (C) 2018-2020 University of Bologna
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
#include "pulp_nn_utils.h"

#define log2(x) __builtin_pulp_fl1(x)
#define min(a,b) ((a)<(b)?(a):(b))

// Taken from PULP-DSP - Moritz Scherer is original author
void plp_sqrt_q64(const int64_t *__restrict__ pSrc,
		  const uint64_t fracBits,
		  int64_t *__restrict__ pRes) {

    int64_t number = *pSrc;
    int64_t root = 0;

    int64_t start = 0;
    int64_t end = 46342; // smallest integer that is larger than sqrt(0x7FFFFFFF)
    int64_t mid;

    if (number > 0) {

        while (start <= end) {

            mid = (start + end) >> 1;

            if (((mid * mid) >> fracBits) == number) {
                root = mid;
                break;
            }

            if (((mid * mid) >> fracBits) < number) {
                start = mid + 1;
                root = mid;
            } else {
                end = mid - 1;
            }
        }

        *pRes = root;

    } else {
        *pRes = 0;
    }
}

void __attribute__ ((noinline))  pulp_nn_layernorm_i8_i8 (
    int8_t * Im_in,              // pointer to the input 
    int8_t * Im_out,             // pointer to the output
    int32_t * weight,               
    int32_t inputSize,          // total number of elt in the input
    int32_t lastDimLength,     
    int32_t log2D,               // right shift
    uint16_t  dim_im_in_h,
    uint16_t  dim_im_in_w,
    uint16_t  ch_im_in           // number of channels of the IFM
)
{

  int core_id = pi_core_id();
  int Log2Core = log2(NUM_CORES);

  int row_dim = inputSize/lastDimLength;
  int row_per_core = (row_dim >> Log2Core) + ((row_dim & (NUM_CORES-1))!=0);

  int start_row, stop_row;
  start_row = min(row_per_core * core_id, row_dim);
  stop_row = min(start_row + row_per_core, row_dim);
  
  int64_t mean;
  int64_t sum;
  int64_t std;
  int64_t temp;
  uint16_t biasOffset = lastDimLength; //Bias are packed in the weight variable with the actual weights

  for(int i = start_row; i < stop_row; i++){
    sum = 0;
    mean = 0;
    for (int j=0;j<lastDimLength; j++){
      mean += Im_in[j + i*lastDimLength];
    }
    mean = mean / lastDimLength;
    for (int j=0;j<lastDimLength; j++){
      temp = (Im_in[j + i*lastDimLength] - mean);
      sum += temp*temp;
    }
    sum = sum / lastDimLength;
    sum += 1;
    plp_sqrt_q64(&sum, 0, &std);
    
    for (int j=0;j<lastDimLength; j++){
      Im_out[j + i*lastDimLength] = (((((int64_t)Im_in[j + i*lastDimLength])-mean)*weight[j])/(std) + weight[biasOffset + j]) >> log2D;
    }
  }

  pi_cl_team_barrier(0);
}
