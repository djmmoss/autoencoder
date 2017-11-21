//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef AUTO_ENC_H_
#define AUTO_ENC_H_

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"

#define W_WIDTH 16
#define I_WIDTH 5
typedef ap_fixed<W_WIDTH,I_WIDTH,AP_RND, AP_SAT> interface_t;

typedef ap_fixed<32,10,AP_RND,AP_SAT> accum_t;
typedef ap_fixed<16,7,AP_RND,AP_SAT> weight_t;
typedef ap_fixed<16,7,AP_RND,AP_SAT> bias_t;

typedef ap_fixed<16,7,AP_RND,AP_SAT> layer1_t;
typedef ap_fixed<16,7,AP_RND,AP_SAT> layer2_t;
typedef ap_fixed<16,7,AP_RND,AP_SAT> layer3_t;
typedef ap_fixed<16,7,AP_RND,AP_SAT> layer4_t;
typedef ap_fixed<16,7,AP_RND,AP_SAT> layer5_t;

/*
typedef float interface_t;
typedef float accum_t;
typedef float weight_t;
typedef float bias_t;

typedef float layer1_t;
typedef float layer2_t;
typedef float layer3_t;
typedef float layer4_t;
typedef float layer5_t;

*/
typedef struct comp {
    interface_t re;
    interface_t im;
} cplx;

short float2short(float f, int i_width);
float short2float(short s, int i_width);
/*
short fxd2short(interface_t val);
interface_t short2fxd(short val);
*/
template <int N_IN>
void fxd2short_stream(hls::stream<interface_t> &in, hls::stream<short> &out);

template <int N_IN>
void short2fxd_stream(hls::stream<short> &in, hls::stream<interface_t> &out);

// Prototype of top level function for C-synthesis
void auto_enc(
      hls::stream<cplx> &data,
      hls::stream<interface_t> &res,
      unsigned short &const_size_in,
      unsigned short &const_size_out,
      unsigned int rd_addr,
      unsigned int wr_addr,
      interface_t wr_val,
      interface_t rd_val);

#endif
