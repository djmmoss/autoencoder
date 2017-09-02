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

// AXI-Stream port type is compatible with pointer, reference, & array input / ouputs only
// See UG902 Vivado High Level Synthesis guide (2014.4) pg 157 Figure 1-49
void auto_enc(
      hls::stream<input_t> &data,
      hls::stream<result_t> &res,
      unsigned short &const_size_in,
      unsigned short &const_size_out)
{
    // Remove ap ctrl ports (ap_start, ap_ready, ap_idle, etc) since we only use the AXI-Stream ports
    #pragma HLS INTERFACE ap_ctrl_none port=return

    // Connect size indicators
    #pragma HLS INTERFACE ap_none port=const_size_in
    #pragma HLS INTERFACE ap_none port=const_size_out
    const_size_in   = N_LAYER_IN;
    const_size_out  = N_LAYER_OUT;

    // ****************************************
    // NETWORK INSTATIATION
    // ****************************************

    // Layer 1
    hls::stream<layer1_t> a_l1, h_l1;
    nnet::compute_layer<input_t, layer1_t, weight_t, bias_t, accum_t, N_LAYER_IN, 32>(data, a_l1, w_l1, b_l1);
    nnet::relu<layer1_t, layer1_t, 32>(a_l1, h_l1);

    // Layer 2
    hls::stream<layer2_t> a_l2, h_l2;
    nnet::compute_layer<layer1_t, layer2_t, weight_t, bias_t, accum_t, 32, 16>(h_l1, a_l2, w_l2, b_l2);
    nnet::relu<layer2_t, layer2_t, 16>(a_l2, h_l2);

    // Layer 3
    hls::stream<layer3_t> a_l3, h_l3;
    nnet::compute_layer<layer2_t, layer3_t, weight_t, bias_t, accum_t, 16, 8>(h_l2, a_l3, w_l3, b_l3);
    nnet::relu<layer3_t, layer3_t, 8>(a_l3, h_l3);

    // Layer 4
    hls::stream<layer4_t> a_l4, h_l4;
    nnet::compute_layer<layer3_t, layer4_t, weight_t, bias_t, accum_t, 8, 8>(h_l3, a_l4, w_l4, b_l4);
    nnet::relu<layer4_t, layer4_t, 8>(a_l4, h_l4);

    // Layer 5
    hls::stream<layer5_t> a_l5, h_l5;
    nnet::compute_layer<layer4_t, layer5_t, weight_t, bias_t, accum_t, 8, 16>(h_l4, a_l5, w_l5, b_l5);
    nnet::relu<layer5_t, layer5_t, 16>(a_l5, h_l5);

    // Layer 6
    nnet::compute_layer<layer5_t,result_t, weight_t, bias_t, accum_t, 16, 32>(h_l5, res, w_l6, b_l6);
}
