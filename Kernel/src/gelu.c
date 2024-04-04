/*
 * pulp_nn_GELU.c
 * Nazareno Bruschi <nazareno.bruschi@unibo.it>
 * Angelo Garofalo <angelo.garofalo@unibo.it>
 * Victor Jung <jungvi@iis.ee.ethz.ch>
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

#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))

void __attribute__ ((noinline))  pulp_nn_gelu_i8_i8 (
    int8_t * Im_in,              // pointer to the input 
    int8_t * Im_out,             // pointer to the output            
    int32_t b,                   // total number of elt in the input   
    int32_t log2D,               // right shift
    int32_t one,
    int32_t totScaler,
    int32_t dataSize,
    uint16_t  dim_im_in_h,
    uint16_t  dim_im_in_w,
    uint16_t  ch_im_in           // number of channels of the IFM
)
{
    int core_id = pi_core_id();
    int n_cores = NUM_CORES;
    if (dim_im_in_h < NUM_CORES)
    {
      n_cores = dim_im_in_h;
    }
    int  Log2Core = log2(n_cores);
    uint8_t extra_chunk = ((dim_im_in_w & (NUM_CORES-1)) != 0);
    uint8_t extra_chunk_r;
    uint16_t dim_out_x_r;
    uint8_t section;
    int core_id_r;

    if(extra_chunk && dim_im_in_w > 1)
    {
      Log2Core = log2(NUM_CORES >> 1);
      core_id_r = (core_id >> 1);
      dim_out_x_r = (dim_im_in_w >> 1);
      section = (core_id & 0x1);
      extra_chunk_r = ((dim_im_in_h & ((NUM_CORES >> 1) - 1)) != 0);
    }
    else
    {
      Log2Core = log2(NUM_CORES);
      core_id_r = core_id;
      dim_out_x_r = dim_im_in_w;
      section = 0;
      extra_chunk_r = extra_chunk;
      extra_chunk = 0;
    }
    int chunk = (dim_im_in_h >> Log2Core) + extra_chunk_r;

    int16_t sign, x, x_abs, q;
    int8_t d;
    int32_t L, y;

    for(int i=0; i<dataSize; i++){
      x = Im_in[i];
      sign = (x > 0) - (x < 0); // sgn(x)
      x_abs = sign*x; // abs(x)
      if (x_abs > -b) {
        q = -b;
      } else {
        q = x_abs;
      }
      d = q + b;
      L = sign * (-(d*d) + one);
      y = ((x * (one + L))>>1);
      Im_out[i] = max(min(((int32_t)(y*totScaler) >> log2D), 127),-128);
    }

    // int start_pixel = min((chunk * core_id_r), dim_im_in_h);
    // int stop_pixel = min(start_pixel + chunk, dim_im_in_h);
    // int8_t *pOutBuffer = Im_out + (start_pixel * ch_im_in * dim_im_in_h) + (section * ch_im_in * dim_im_in_w);
    // int8_t *target =  Im_in + (start_pixel * ch_im_in * dim_im_in_h) + (section * ch_im_in * dim_im_in_w);
    // int16_t *wei = emb + (start_pixel * ch_im_in * dim_im_in_h) + (section * ch_im_in * dim_im_in_w);
    // for (int spatial = 0; spatial < dim_im_in_w*ch_im_in*(stop_pixel-start_pixel); spatial+=1)
    // {
    //    int8_t intermediate =  pulp_nn_requantshift_i8_i8((int32_t)*target, mul1, add1, div1);
    //    int32_t embedded = (int32_t)intermediate + *wei;
    //    *pOutBuffer = (int8_t) pulp_nn_requantshift_i8_i8(embedded, mul2, add2, div2);
    //    target += 1;
    //    pOutBuffer += 1;
    //    wei +=1;
    // }
    pi_cl_team_barrier(0);
}
