#include "../inc/thorir_dma.h"
#include "pmsis.h"

#define MIN(a,b) ((a)<(b)?(a):(b))
// Does not support multicore DMA yet
#define SINGLE_CORE_DMA

void thorir_hwc_to_chw(DMA_copy *copy){
#ifdef SINGLE_CORE_DMA
  if (pi_core_id() == 0) {
#endif
  int start_pixel, stop_pixel; // "pixel" is a misnomer; the CHANNELS are divided between the cores
  // this function assumes that a DW tile is always as wide as the complete feature map (this is enforced by DORY's tiler)
  // if there is only 1 DMA control unit for the cluster (e.g., Kraken), we can't execute DMA calls on multiple clusters.
#ifndef SINGLE_CORE_DMA
  int core_id = pi_core_id();
  int Log2Core = log2(NUM_CORES);
  int number_of_copies_per_core = (copy->length_1d_copy >> Log2Core) + ((copy->length_1d_copy & (NUM_CORES-1))!=0);
  start_pixel = MIN(number_of_copies_per_core * core_id, copy->length_1d_copy);
  stop_pixel = MIN(start_pixel + number_of_copies_per_core, copy->length_1d_copy);
#else
  start_pixel = 0;
  stop_pixel = copy->length_1d_copy;
#endif
  void * loc = copy->loc + copy->number_of_1d_copies*copy->number_of_2d_copies*start_pixel;
  void * ext = copy->ext + start_pixel;
  const int size_2d = copy->number_of_1d_copies * copy->number_of_2d_copies;

  for (int i=start_pixel; i<stop_pixel; i++) {
    pi_cl_dma_copy_2d_t dma_copy;
    dma_copy.dir = copy->dir;
    dma_copy.size = size_2d;
    dma_copy.ext = ext;
    dma_copy.loc = loc;
    dma_copy.length = 1;
    dma_copy.stride = copy->stride_1d;
    if(copy->dir == 1)
      dma_copy.dir = PI_CL_DMA_DIR_EXT2LOC;
    else
      dma_copy.dir = PI_CL_DMA_DIR_LOC2EXT;
    pi_cl_dma_memcpy_2d(&dma_copy);
    pi_cl_dma_wait(&dma_copy);
    // pi_cl_team_barrier(0);
    ext += 1; // next channel
    loc += copy->number_of_1d_copies * copy->number_of_2d_copies;
  }
#ifdef SINGLE_CORE_DMA
  }
#endif
}


void thorir_1d(DMA_copy *copy) {
  if (pi_core_id() == 0) {
    pi_cl_dma_copy_t dma_copy;
    dma_copy.ext = copy->ext;
    dma_copy.loc = copy->loc;
    dma_copy.dir = copy->dir;
    dma_copy.size = copy->length_1d_copy * copy->number_of_1d_copies * copy->number_of_2d_copies;
    dma_copy.merge = 0;
    //dma_copy.dir = (copy->dir == 1) ? PI_CL_DMA_DIR_EXT2LOC : PI_CL_DMA_DIR_LOC2EXT;
    if(copy->dir == 1)
        dma_copy.dir = PI_CL_DMA_DIR_EXT2LOC;
    else
        dma_copy.dir = PI_CL_DMA_DIR_LOC2EXT;

    pi_cl_dma_memcpy(&dma_copy);
    pi_cl_dma_wait(&dma_copy);
  }
}

void thorir_2d(DMA_copy *copy) {
  if (pi_core_id() == 0) {
    pi_cl_dma_copy_2d_t dma_copy;
    const int size_2d = copy->number_of_1d_copies * copy->length_1d_copy * copy->number_of_2d_copies;
    const int stride = (copy->number_of_2d_copies == 1) ? copy->stride_1d : copy->stride_2d;
    const int size_1d = (copy->number_of_2d_copies == 1) ? copy->length_1d_copy : copy->length_1d_copy * copy->number_of_1d_copies;
    dma_copy.dir = copy->dir;
    dma_copy.size = size_2d;
    dma_copy.ext = copy->ext;
    dma_copy.loc = copy->loc;
    dma_copy.length = size_1d;
    dma_copy.stride = stride;

    if(copy->dir == 1)
        dma_copy.dir = PI_CL_DMA_DIR_EXT2LOC;
    else
        dma_copy.dir = PI_CL_DMA_DIR_LOC2EXT;
    pi_cl_dma_memcpy_2d(&dma_copy);
    pi_cl_dma_wait(&dma_copy);
  }
}

void thorir_3d(DMA_copy *copy) {
#ifdef SINGLE_CORE_DMA
  if (pi_core_id() == 0) {
#endif
  int start_pixel, stop_pixel;
#ifndef SINGLE_CORE_DMA
  int core_id = pi_core_id();
  int Log2Core = log2(NUM_CORES);
  int number_of_2d_copies_per_core = (copy->number_of_2d_copies >> Log2Core) + ((copy->number_of_2d_copies & (NUM_CORES-1))!=0);
  start_pixel = MIN(number_of_2d_copies_per_core * core_id, copy->number_of_2d_copies);
  stop_pixel = MIN(start_pixel + number_of_2d_copies_per_core, copy->number_of_2d_copies);
#else
  start_pixel = 0;
  stop_pixel = copy->number_of_2d_copies;
#endif
  void *ext = copy->ext + copy->stride_2d*start_pixel;
  void *loc = copy->loc + copy->length_1d_copy*copy->number_of_1d_copies*start_pixel;
  const int size_2d = copy->number_of_1d_copies * copy->length_1d_copy;
  for (int i = start_pixel; i < stop_pixel; i++) {
    pi_cl_dma_copy_2d_t dma_copy;
    dma_copy.dir = copy->dir;
    dma_copy.size = size_2d;
    dma_copy.ext = ext;
    dma_copy.loc = loc;
    dma_copy.length = copy->length_1d_copy;
    dma_copy.stride = copy->stride_1d;
    if(copy->dir == 1)
      dma_copy.dir = PI_CL_DMA_DIR_EXT2LOC;
    else
      dma_copy.dir = PI_CL_DMA_DIR_LOC2EXT;
    pi_cl_dma_memcpy_2d(&dma_copy);
    pi_cl_dma_wait(&dma_copy);
    // pi_cl_team_barrier(0);
    loc += size_2d;
    ext += copy->stride_2d;
  }
#ifdef SINGLE_CORE_DMA
  }
#endif
}


void thorir_dma(DMA_copy *copy) {
  if (copy->hwc_to_chw == 1) {
    thorir_hwc_to_chw(copy);
  }
  else if ((copy->number_of_2d_copies == 1 && copy->number_of_1d_copies == 1) || (copy->stride_1d == copy->length_1d_copy &&  copy->number_of_1d_copies * copy->length_1d_copy == copy->stride_2d) || (copy->number_of_2d_copies == 1 && copy->length_1d_copy == copy->stride_1d)) {
    thorir_1d(copy);
  } else if ((copy->number_of_2d_copies == 1) || (copy->length_1d_copy == copy->stride_1d)) {// wrong!
    thorir_2d(copy);
  } else {
    thorir_3d(copy);
  }
}

