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


typedef ap_fixed<32,10> accum_t;
typedef ap_fixed<32,10> input_t;
typedef ap_fixed<32,10> result_t;
typedef ap_fixed<32,10> weight_t;
typedef ap_fixed<32,10> bias_t;

typedef ap_fixed<32,10> layer1_t;
typedef ap_fixed<32,10> layer2_t;
typedef ap_fixed<32,10> layer3_t;
typedef ap_fixed<32,10> layer4_t;
typedef ap_fixed<32,10> layer5_t;


/*
typedef float accum_t;
typedef float input_t;
typedef float result_t;
typedef float weight_t;
typedef float bias_t;

typedef float layer1_t;
typedef float layer2_t;
typedef float layer3_t;
typedef float layer4_t;
typedef float layer5_t;
*/

#define N_LAYER_IN 32
#define N_LAYER_OUT 32

// Prototype of top level function for C-synthesis
void auto_enc(
      hls::stream<input_t> &data,
      hls::stream<result_t> &res,
      unsigned short &const_size_in,
      unsigned short &const_size_out);


#endif
