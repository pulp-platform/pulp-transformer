/* ----------------------------------------------------------------------
#
# File: linearProjQKVTemplate.c
#
# Last edited: 30.08.2023
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


// #include "../inc/${testInputHeaderName}.h    "

#include "pmsis.h"
#include "bsp/fs.h"
#include "bsp/bsp.h"
#include <bsp/flash/spiflash.h>
#include <bsp/fs/readfs.h>

// #include "../inc/dory.h"
#include "../inc/thorir_dma.h"
#include "../inc/pulp_nn_utils.h"
#include "../inc/pulp_nn_kernels.h"

#define FLASH_BUFF_SIZE 128

#define SLAVE_STACK_SIZE 2048
#define STACK_SIZE      2048

// #define TEST_INPUTS
#define PROFILING
#define GPIO

#ifdef GPIO
  unsigned int GPIOs = 89;
  #define WRITE_GPIO(x) pi_gpio_pin_write(GPIOs,x)
#endif

#ifdef PROFILING
  #define START_PROFILING()  if(pi_core_id()==0){ pi_perf_conf((1<<PI_PERF_ACTIVE_CYCLES)); pi_perf_reset(); pi_perf_stop(); pi_perf_start();}
  #define STOP_PROFILING(str)  if(pi_core_id()==0){ pi_perf_stop(); printf("%s: %d\n", #str, pi_perf_read(PI_PERF_ACTIVE_CYCLES));}
#else
  #define START_PROFILING()
  #define STOP_PROFILING()
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

  char *base = I;
  
  START_PROFILING();
  #ifdef GPIO
  WRITE_GPIO(1);
  #endif

  for(int i=0; i < 2; i++){
    linearQK_4x2_H(I, W, B, O, ${S}, ${E}, ${P}, ${H}, ${requantDiv}, ${requantMul});
    pi_cl_team_barrier(0);
  }

  linearV_4x2_H(I, W, B, O, ${S}, ${E}, ${P}, ${H}, ${requantDiv}, ${requantMul});
  pi_cl_team_barrier(0);

  I = base;
  W = base + ${S*P};
  O = base + ${S*P} + ${S*P};
  for(int i=0; i < ${H}; i++){
    matmulSoftmax_4x2_S(I, W, O, ${S}, ${P}, 1, ${requantDiv}, ${requantMul}, 1, 7, 24, 5, 256);
    pi_cl_team_barrier(0);
  }
  
  I = base;
  W = base + ${S*S};
  O = base + ${S*S} + ${S*P};
  for(int i=0; i < ${H}; i++){
    matmul_4x2_S(I, W, O, ${S}, ${P}, 1, ${requantDiv}, ${requantMul});
    pi_cl_team_barrier(0);
  }

  I = base;
  W = base + ${H*S*P};
  O = base + ${H*S*P} + ${H*S*E};
  linearO_4x2_H(I, W, B, O, ${S}, ${E}, ${P}, ${H}, ${requantDiv}, ${requantMul});
  pi_cl_team_barrier(0);

  #ifdef GPIO
  WRITE_GPIO(0);
  #endif
  STOP_PROFILING(Kernel Execution);
}

void kernel_task(void *task_args) {

  char* L1_buffer = pi_cl_l1_malloc((void *) 0, (uint32_t) ${l1BufferSize});

  // Create L1 tensor pointers
  printf("Declare Buffer Pointers: ");
  char *A = (char *) (L1_buffer + ${dmaTransferSize});
  char *Bias = (char *) (L1_buffer + ${dmaTransferSize} + ${int(sizeBias + dmaTransferSize)});
  char *B = (char *) (L1_buffer + ${dmaTransferSize} + ${int(sizeBias + sizeA + 2*dmaTransferSize)});
  char *O = (char *) (L1_buffer + ${dmaTransferSize} + ${int(sizeBias + sizeA + sizeB + 3*dmaTransferSize)});
  printf("DONE\n");
  
   // Build agrs to give to cluster
  unsigned int args[10] = {
    A,
    B,
    Bias,
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

  #ifdef GPIO
  pi_pad_function_set(GPIOs, 1);
  pi_gpio_pin_configure(GPIOs, PI_GPIO_OUTPUT);
  pi_gpio_pin_write(GPIOs, 0);
  WRITE_GPIO(0);
  #endif

  pi_cluster_conf_init(&conf);
  conf.id=0;
  conf.cc_stack_size = STACK_SIZE;
  printf("DONE\n");

  printf("Allocate L2: ");
  L2_buffer = pi_l2_malloc((uint32_t) ${l2BufferSize});
  printf("DONE\n");

  unsigned int empty_args[0] = {};

  // Start cluster job
  printf("Start Cluster Task");
  // Prepare Task
  pi_cluster_task(&cluster_task, kernel_task, empty_args);
  pi_cluster_task_stacks(&cluster_task, NULL, SLAVE_STACK_SIZE);

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