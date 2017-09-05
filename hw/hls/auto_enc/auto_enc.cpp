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

#include "auto_enc.h"
#include "nnet_layer.h"
#include "nnet_activation.h"

// Network
#include "data/net.h"

// Biases
#include "data/b_l1.h"
#include "data/b_l2.h"
#include "data/b_l3.h"
#include "data/b_l4.h"
#include "data/b_l5.h"
#include "data/b_l6.h"

// Weights
#include "data/w_l1.h"
#include "data/w_l2.h"
#include "data/w_l3.h"
#include "data/w_l4.h"
#include "data/w_l5.h"
#include "data/w_l6.h"

short float2short(float f) {
	short s = f * (1 << 15);

	if (f > 1.0)
		s = 0x7FFF;
	else if (f < -1.0)
		s = 0x8000;
	return s;
}

float short2float(short s, int i_width) {
	float f = (float) s / (1 << (15 - (i_width-1)));
	return f;
}

short fxd2short(interface_t val) {
	short s = (val.range(W_WIDTH-1,0) << (16 - W_WIDTH));
	return s;
}

interface_t short2fxd(short val) {
	interface_t v;
	v(W_WIDTH-1,0) = (val >> (15 - W_WIDTH + I_WIDTH));
	return v;
}

template <int N_IN>
void fxd2short_stream(hls::stream<interface_t> &in, hls::stream<short> &out) {
    for (int ii = 0; ii < N_IN; ii++) {
        #pragma UNROLL
        out << fxd2short(in.read());
    }
}

template <int N_IN>
void short2fxd_stream(hls::stream<short> &in, hls::stream<interface_t> &out) {
    for (int ii = 0; ii < N_IN; ii++) {
        #pragma UNROLL
        out << short2fxd(in.read());
    }
}

// AXI-Stream port type is compatible with pointer, reference, & array input / ouputs only
// See UG902 Vivado High Level Synthesis guide (2014.4) pg 157 Figure 1-49
void auto_enc(
      hls::stream<short> &data,
      hls::stream<short> &res,
      unsigned short &const_size_in,
      unsigned short &const_size_out)
{
    // Remove ap ctrl ports (ap_start, ap_ready, ap_idle, etc) since we only use the AXI-Stream ports
    #pragma HLS INTERFACE ap_ctrl_none port=return

    #pragma HLS DATAFLOW

    // Connect size indicators
    #pragma HLS INTERFACE ap_none port=const_size_in
    #pragma HLS INTERFACE ap_none port=const_size_out
    const_size_in   = LAYER_1;
    const_size_out  = LAYER_1;

    // ****************************************
    // NETWORK INSTATIATION
    // ****************************************

    // Input Conversion
    hls::stream<interface_t> h_l0;
    short2fxd_stream<LAYER_1>(data, h_l0);

    // Layer 1
    hls::stream<layer1_t> a_l1, h_l1;
    nnet::compute_layer<interface_t, layer1_t, weight_t, bias_t, accum_t, LAYER_1, LAYER_1>(h_l0, a_l1, w_l1, b_l1);
    nnet::relu<layer1_t, layer1_t, LAYER_1>(a_l1, h_l1);

    // Layer 2
    hls::stream<layer2_t> a_l2, h_l2;
    nnet::compute_layer<layer1_t, layer2_t, weight_t, bias_t, accum_t, LAYER_1, LAYER_2>(h_l1, a_l2, w_l2, b_l2);
    nnet::relu<layer2_t, layer2_t, LAYER_2>(a_l2, h_l2);

    // Layer 3
    hls::stream<layer3_t> a_l3, h_l3;
    nnet::compute_layer<layer2_t, layer3_t, weight_t, bias_t, accum_t, LAYER_2, LAYER_3>(h_l2, a_l3, w_l3, b_l3);
    nnet::relu<layer3_t, layer3_t, LAYER_3>(a_l3, h_l3);

    // Layer 4
    hls::stream<layer4_t> a_l4, h_l4;
    nnet::compute_layer<layer3_t, layer4_t, weight_t, bias_t, accum_t, LAYER_3, LAYER_3>(h_l3, a_l4, w_l4, b_l4);
    nnet::relu<layer4_t, layer4_t, LAYER_3>(a_l4, h_l4);

    // Layer 5
    hls::stream<layer5_t> a_l5, h_l5;
    nnet::compute_layer<layer4_t, layer5_t, weight_t, bias_t, accum_t, LAYER_3, LAYER_2>(h_l4, a_l5, w_l5, b_l5);
    nnet::relu<layer5_t, layer5_t, LAYER_2>(a_l5, h_l5);

    // Layer 6
    hls::stream<interface_t> h_l6;
    nnet::compute_layer<layer5_t, interface_t, weight_t, bias_t, accum_t, LAYER_2, LAYER_1>(h_l5, h_l6, w_l6, b_l6);

    // Output Conversion
    fxd2short_stream<LAYER_1>(h_l6, res);

}
