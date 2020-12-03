/****************************************************************
 * Copyright (c) 2020~2020, 18-643 Course Staff, CMU
 * All rights reserved.

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:

 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.

 * 2. Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.

 * The views and conclusions contained in the software and
 * documentation are those of the authors and should not be
 * interpreted as representing official policies, either expressed or
 * implied, of the FreeBSD Project.
 ****************************************************************/

/****************************************************************
 * Blocked (without copying) convolution layer implementation 
 * based on Figure 5:
 *    C. Zhang, et al., "Optimizing FPGA-based Accelerator 
 *    Design for Deep Convolutional Neural Networks," FPGA, 2015.
 ****************************************************************/

#include "../host/inc/util643.h"
#include "../host/inc/instance643.h"
#include "../host/inc/kernel643.h"

__attribute((reqd_work_group_size(1, 1, 1)))
__kernel void cnn(__global const cnndata_t* restrict input, __global const cnndata_t* restrict weights, __global cnndata_t* restrict output, 
                  const uint64_t batch_size,  const kernel_size kernel_params, const layer_size layer_params)
{
  uint64_t iter;
  uint64_t row, col, to, ti;

  uint64_t _K_wts = FIX_K ? K_WTS : layer_params.K_wts;
  uint64_t _S_wts = FIX_S ? S_WTS : layer_params.S_wts;
  
  uint64_t _R_ofm = FIX_R ? R_OFM : layer_params.R_ofm;
  uint64_t _C_ofm = FIX_C ? C_OFM : layer_params.C_ofm;
  uint64_t _M_ofm = FIX_M ? M_OFM : layer_params.M_ofm;

  uint64_t _R_ifm = (_R_ofm * _S_wts + _K_wts - _S_wts);
  uint64_t _C_ifm = (_C_ofm * _S_wts + _K_wts - _S_wts);
  uint64_t _N_ifm = FIX_N ? N_IFM : layer_params.N_ifm;
  
  uint64_t _Tr = FIX_TR ? TR : kernel_params.Tr;
  uint64_t _Tc = FIX_TC ? TC : kernel_params.Tc;
  uint64_t _Tm = FIX_TM ? TM : kernel_params.Tm;
  uint64_t _Tn = FIX_TN ? TN : kernel_params.Tn;
 
  for(iter = 0; iter < batch_size; iter++) {
    
    for(row = 0; row < _R_ofm; row += _Tr) {
      for(col = 0; col < _C_ofm ; col += _Tc) {
        for(to = 0; to < _M_ofm; to += _Tm) {
          for(ti = 0; ti < _N_ifm; ti += _Tn) {
            uint64_t trr, tcc, too, tii;
  
            for(trr = row; trr < MIN(row + _Tr, _R_ofm); trr++){
              for(tcc = col; tcc < MIN(col + _Tc, _C_ofm); tcc++){
                for(too = to; too < MIN(to + _Tm, _M_ofm); too++) {    
                  for(tii = ti; tii < MIN(ti + _Tn, _N_ifm); tii++) { 
                    uint64_t i, j;
                    for(i = 0; i < _K_wts; i++){
                      for(j = 0; j < _K_wts; j++){
                        ARRAYo(output, iter, too, trr, tcc, batch_size, _M_ofm, _R_ofm, _C_ofm)+=
                          ARRAYw(weights, too, tii, i, j, _M_ofm, _N_ifm, _K_wts, _K_wts) *
                          ARRAYi(input, iter, tii, _S_wts * trr + i, _S_wts * tcc + j, 
                            batch_size, _N_ifm, _R_ifm, _C_ifm);
                      }
                    }
                  }
                }
              }
            }
          } 
        } 
      } 
    }
  }
}
