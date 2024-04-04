/* ----------------------------------------------------------------------
#
# File: linearProjQKVTemplate.c
#
# Last edited: 14.09.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author:
# - Victor Jung, jungvi@iis.ee.ethz.ch, ETH Zurich
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


#include "../inc/${testInputHeaderName}.h"

#include "pmsis.h"
#include "bsp/fs.h"
#include "bsp/bsp.h"
#include <bsp/flash/spiflash.h>
#include <bsp/fs/readfs.h>

// #include "../inc/dory.h"
#include "../inc/thorir_dma.h"
#include "../inc/pulp_nn_utils.h"
#include "../inc/pulp_nn_kernels.h"

#define clip8(x) __builtin_pulp_clip_r(x, 127)

#define FLASH_BUFF_SIZE 128

// #define TEST_INPUTS
// #define PROFILING

#ifdef PROFILING
  #define START_PROFILING(){\
      if(pi_core_id()==0){\
        pi_perf_conf(1<<${perf_counter});\
        pi_perf_start();\
      }\
    }

  #define STOP_PROFILING(){\
    if(pi_core_id()==0){\
      pi_perf_stop();\
      printf("Kernel Execution: %d\n", pi_perf_read(PI_PERF_ACTIVE_CYCLES));\
      printf("${perf_counter}:%d\n", pi_perf_read(${perf_counter}));\
    }\
  }

#else
  #define START_PROFILING(){}
  #define STOP_PROFILING(){}
#endif


struct pi_mx25u51245g_conf flash_conf;
static struct pi_hyper_conf ram_conf;
static struct pi_device ram;
static int activations_input;
static uint8_t flashBuffer[FLASH_BUFF_SIZE];

void cluster_fork(void *args) {

  // Unpack args
  char *I = ((char **)args)[0];
  char *W = ((char **)args)[1];
  char *B = ((char **)args)[2]; 
  char *O = ((char **)args)[3];
  int S = ((int *)args)[4];
  int E = ((int *)args)[5];
  int P = ((int *)args)[6];
  int H = ((int *)args)[7];
  int requantDiv = ((int *)args)[8];
  int requantMul = ((int *)args)[9];

  // DMA Transfer Inputs from L2 to L1
  printf("Allocate DMA Channel: ");
  volatile DMA_copy DMA_copy_I, DMA_copy_W, DMA_copy_B;
  printf("DONE\n");

  DMA_copy_I.hwc_to_chw = 0;
  DMA_copy_I.stride_2d = 0;
  DMA_copy_I.stride_1d = 0;
  DMA_copy_I.dir = 1;
  DMA_copy_I.ext = &${inputVectorName};
  DMA_copy_I.loc = I;
  DMA_copy_I.number_of_2d_copies = 1;
  DMA_copy_I.number_of_1d_copies = 1;
  DMA_copy_I.length_1d_copy = ${dmaTransferSize};

  DMA_copy_W.hwc_to_chw = 0;
  DMA_copy_W.stride_2d = 0;
  DMA_copy_W.stride_1d = 0;
  DMA_copy_W.dir = 1;
  DMA_copy_W.ext = &${weightVectorName};
  DMA_copy_W.loc = W;
  DMA_copy_W.number_of_2d_copies = 1;
  DMA_copy_W.number_of_1d_copies = 1;
  DMA_copy_W.length_1d_copy = ${dmaTransferSize};

  DMA_copy_B.hwc_to_chw = 0;
  DMA_copy_B.stride_2d = 0;
  DMA_copy_B.stride_1d = 0;
  DMA_copy_B.dir = 1;
  DMA_copy_B.ext = &${biasVectorName};
  DMA_copy_B.loc = B;
  DMA_copy_B.number_of_2d_copies = 1;
  DMA_copy_B.number_of_1d_copies = 1;
  DMA_copy_B.length_1d_copy = ${biasSize};

  // START_PROFILING();
  thorir_dma(DMA_copy_I);
  pi_cl_team_barrier(0);

  for(int i = 0; i < ${numberOfInputTransfer}; i++){
    DMA_copy_I.ext = (int)&(${inputVectorName}) + ${dmaTransferSize}*i;
    DMA_copy_I.loc = I + ${dmaTransferSize}*i;
    thorir_dma(DMA_copy_I);
    pi_cl_team_barrier(0); 
  }

  for(int i = 0; i < ${numberOfWeightTransfer}; i++){
    DMA_copy_W.ext = (int)&(${weightVectorName}) + ${dmaTransferSize}*i;
    DMA_copy_W.loc = W + ${dmaTransferSize}*i;
    thorir_dma(DMA_copy_W);
    pi_cl_team_barrier(0); 
  }
  
  thorir_dma(DMA_copy_B);
  pi_cl_team_barrier(0);
  // STOP_PROFILING();

  #ifdef TEST_INPUTS
    if (pi_core_id()==0) {
      printf("Input check:\n");
      int inputCheck = 1;
      for(int i = 0; i < ${weightSize}; i++){
        if((int8_t)W[i] != ${weightVectorName}[i]){
          printf("\nError: Input Transfer Error at %d: %d / %d\n", i, I[i], ${inputVectorName}[i]);
          inputCheck = 0;
        }
      }
      if(inputCheck){
        printf("Input Transfer Correct");
      }
      printf("\n");

      printf("Weight check:\n");
      int weightCheck = 1;
      for(int i = 0; i < ${weightSize}; i++){
        if((int8_t)W[i] != ${weightVectorName}[i]){
          printf("\nError: Weight Transfer Error at %d: %d / %d\n", i, W[i], ${weightVectorName}[i]);
          weightCheck = 0;
        }
      }
      if(weightCheck){
        printf("Weight Transfer Correct");
      }
      printf("\n");

      printf("Bias check:\n");
      int biasCheck = 1;
      int16_t *B16 = (int16_t *)B;
      for(int i = 0; i < ${int(biasSize/2)}; i++){
        if((int16_t)B16[i] != ${biasVectorName}[i]){
          printf("\nError: Bias Transfer Error at %d: %d / %d\n", i, (int16_t)B16[i], ${biasVectorName}[i]);
          biasCheck = 0;
        }
      }
      if(biasCheck){
        printf("Weight Transfer Correct");
      }
      printf("\n");
    }
  #endif
  
  pi_cl_team_barrier(0);
  START_PROFILING();
  ${kernelName}(I, W, B, O, ${S}, ${E}, ${P}, ${H}, ${requantDiv}, ${requantMul});
  pi_cl_team_barrier(0);
  STOP_PROFILING();
  
  if (pi_core_id()==0) {
    printf("Output:\n");
    for(int i = 0; i < ${outputSize}; i++){
      printf("%d, ", (int8_t)O[i]);
    }
    printf("\n");
  }
}

void kernel_task(void *task_args) {

  char* L1_buffer = pi_cl_l1_malloc((void *) 0, (uint32_t) ${l1BufferSize});

  // Create L1 tensor pointers
  printf("Declare Buffer Pointers: ");
  char *I = (char *) (L1_buffer + 0);
  char *W = (char *) (L1_buffer + 0 + ${int(inputSize + dmaTransferSize)});
  char *B = (char *) (L1_buffer + 0 + ${int(inputSize + dmaTransferSize + weightSize + dmaTransferSize)});
  char *O = (char *) (L1_buffer + 0 + ${int(inputSize + dmaTransferSize + weightSize + dmaTransferSize + biasSize + 2*dmaTransferSize)});
  printf("DONE\n");
  
   // Build agrs to give to cluster
  unsigned int args[10] = {
    I,
    W,
    B,
    O,
    ${S},
    ${E},
    ${P},
    ${H},
    ${requantDiv},
    ${requantMul}
  };

  pi_cl_team_fork(NUM_CORES, cluster_fork, args);
  pi_cl_l1_free((void *) 0, L1_buffer, (uint32_t) ${l1BufferSize});
}

int main () {

  char* L1_buffer;
  char* L2_buffer;

  printf("Configure mcu: ");
  struct pi_device cluster_dev = {0};
  struct pi_cluster_conf conf;
  struct pi_cluster_task cluster_task = {0};
  struct pi_device fs;
  struct pi_device flash;
  
  pi_freq_set(PI_FREQ_DOMAIN_FC, ${fcFrequency});
  pi_time_wait_us(10000);
  pi_freq_set(PI_FREQ_DOMAIN_CL, ${clFrequency});
  pi_time_wait_us(10000);

  pi_cluster_conf_init(&conf);
  conf.id=0;
  printf("DONE\n");

  printf("Allocate L2: ");
  L2_buffer = pi_l2_malloc((uint32_t) ${l2BufferSize});
  printf("DONE\n");

  unsigned int empty_args[0] = {};

  // Start cluster job
  printf("Start Cluster Task");
  // Prepare Task
  pi_cluster_task(&cluster_task, kernel_task, empty_args);

  pi_open_from_conf(&cluster_dev, &conf);
  if (pi_cluster_open(&cluster_dev)){
    printf("Error: Can't open cluster\n");
    return -1;
  }

  // Then offload an entry point, this will get executed on the cluster controller
  // cluster_task.stack_size = 3500;
  // cluster_task.slave_stack_size = 3400;
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

  // Close the cluster
  printf("End Cluster Task");
  pi_cluster_close(&cluster_dev);
}