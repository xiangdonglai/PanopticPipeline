/*
 * Copyright (c) 2011. Shohei NOBUHARA, Kyoto University and Carnegie
 * Mellon University. This code may be used, distributed, or modified
 * only for research purposes or under license from Kyoto University or
 * Carnegie Mellon University. This notice must be retained in all copies.
 */
#ifndef PFCMU_CONFIG_H
#define PFCMU_CONFIG_H

#include <endian.h>
#include <stdint.h>

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

namespace PFCMU {
  const static int CAMS = 24;

  /// Max chunk size for Scatter-gathering DMA from PF-CMU to main memory. If an image buffer has too many chunks inside, DMA transmission with it will be very slow. See get_sgdma_length() in libpfcmu/src/capture++.cc .
  const static int MAX_SGDMA_SIZE = 200;

#if defined(PF_USE_OPENCV) || defined(CV_BayerGR2BGR) || defined(__OPENCV_IMGPROC_IMGPROC_C_H__)
  const static int CV_BAYER2BGR = CV_BayerGR2BGR;
#endif

  // use "%llu" for printf.
  // see profusion/lib/libviewplus/include/PF_EZInterface.h
  typedef unsigned long long timestamp_t;

  /** 
   * Get embedded timestamp from the first 4 bytes of the image data
   * 
   * @param data [in] image data
   * 
   * @return timestamp
   */
  inline timestamp_t get_timestamp(const void * data) {
    return be32toh(*(reinterpret_cast<const uint32_t *>(data)));
  }

}

#endif
