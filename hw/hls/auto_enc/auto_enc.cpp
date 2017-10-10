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

/*
short float2short(float f, int i_width) {
	short s = f * (1 << (15 - i_width));
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

template<int N_IN, int N_OUT>
interface_t get_mem_val(weight_t w[N_IN][N_OUT], bias_t b[N_OUT], bool worb, ap_uint<11> loc_1, ap_uint<16> loc_2) {
    short res;
    if (worb) {
        res = fxd2short(w[loc_1][loc_2]);
    } else {
        res = fxd2short(b[loc_2]);
    }
    return res;
}
*/

template<int N_IN, int N_OUT>
void set_mem_val(interface_t val, weight_t w[N_IN][N_OUT], bias_t b[N_OUT], bool worb, ap_uint<11> loc_1, ap_uint<16> loc_2) {
    if (worb) {
        w[loc_1][loc_2] = val;
    } else {
        b[loc_2] = val;
    }
}

template<int L_1, int L_2, int L_3, int L_4, int L_5, int L_6>
void weight_server(
        unsigned int i_wr_addr,
        interface_t i_wr_val,
        unsigned int i_rd_addr,
        interface_t &o_rd_val,
        weight_t o_w_l1[L_1][L_1],
        weight_t o_w_l2[L_1][L_2],
        weight_t o_w_l3[L_2][L_3],
        weight_t o_w_l4[L_3][L_4],
        weight_t o_w_l5[L_4][L_5],
        weight_t o_w_l6[L_5][L_6],
        bias_t o_b_l1[L_1],
        bias_t o_b_l2[L_2],
        bias_t o_b_l3[L_3],
        bias_t o_b_l4[L_4],
        bias_t o_b_l5[L_5],
        bias_t o_b_l6[L_6]
        ) {
    #pragma HLS PIPELINE II=1 enable_flush

    ap_uint<32> wr_addr = i_wr_addr;
    bool wr_weight_or_bias = wr_addr[31]; // Weight (1), Bias (0)
    ap_uint<4> wr_layer_number = wr_addr(31,27);
    ap_uint<11> wr_loc_1 = wr_addr(27, 16);
    ap_uint<16> wr_loc_2 = wr_addr(16, 0);


    if (wr_layer_number == 1) {
        //set_mem_val<L_1, L_1>(i_wr_val, w_l1, b_l1, wr_weight_or_bias, wr_loc_1, wr_loc_2);
    } else if (wr_layer_number == 2) {
        set_mem_val<L_1, L_2>(i_wr_val, w_l2, b_l2, wr_weight_or_bias, wr_loc_1, wr_loc_2);
    } else if (wr_layer_number == 3) {
        set_mem_val<L_2, L_3>(i_wr_val, w_l3, b_l3, wr_weight_or_bias, wr_loc_1, wr_loc_2);
    } else if (wr_layer_number == 4) {
        //set_mem_val<L_3, L_4>(i_wr_val, w_l4, b_l4, wr_weight_or_bias, wr_loc_1, wr_loc_2);
    } else if (wr_layer_number == 5) {
        set_mem_val<L_4, L_5>(i_wr_val, w_l5, b_l5, wr_weight_or_bias, wr_loc_1, wr_loc_2);
    } else if (wr_layer_number == 6) {
        set_mem_val<L_5, L_6>(i_wr_val, w_l6, b_l6, wr_weight_or_bias, wr_loc_1, wr_loc_2);
    }

    /*
    ap_uint<32> rd_addr = i_rd_addr;
    bool rd_weight_or_bias = rd_addr[31]; // Weight (1), Bias (0)
    ap_uint<4> rd_layer_number = rd_addr(31,27);
    ap_uint<11> rd_loc_1 = rd_addr(27, 16);
    ap_uint<16> rd_loc_2 = rd_addr(16, 0);

    if (rd_layer_number == 1) {
        o_rd_val = get_mem_val<L_1, L_1>(w_l1, b_l1, rd_weight_or_bias, rd_loc_1, rd_loc_2);
    } else if (rd_layer_number == 2) {
        o_rd_val = get_mem_val<L_1, L_2>(w_l2, b_l2, rd_weight_or_bias, rd_loc_1, rd_loc_2);
    } else if (rd_layer_number == 3) {
        o_rd_val = get_mem_val<L_2, L_3>(w_l3, b_l3, rd_weight_or_bias, rd_loc_1, rd_loc_2);
    } else if (rd_layer_number == 4) {
        o_rd_val = get_mem_val<L_3, L_4>(w_l4, b_l4, rd_weight_or_bias, rd_loc_1, rd_loc_2);
    } else if (rd_layer_number == 5) {
        o_rd_val = get_mem_val<L_4, L_5>(w_l5, b_l5, rd_weight_or_bias, rd_loc_1, rd_loc_2);
    } else if (rd_layer_number == 6) {
        o_rd_val = get_mem_val<L_5, L_6>(w_l6, b_l6, rd_weight_or_bias, rd_loc_1, rd_loc_2);
    }
    */

    for (int j = 0; j < L_1; j++) {
        for (int i = 0; i < L_1; i++) {
            o_w_l1[i][j] = w_l1[i][j];
        }
        o_b_l1[j] = b_l1[j];
    }

    for (int j = 0; j < L_2; j++) {
        for (int i = 0; i < L_1; i++) {
            o_w_l2[i][j] = w_l2[i][j];
        }
        o_b_l2[j] = b_l2[j];
    }

    for (int j = 0; j < L_3; j++) {
        for (int i = 0; i < L_2; i++) {
            o_w_l3[i][j] = w_l3[i][j];
        }
        o_b_l3[j] = b_l3[j];
    }

    for (int j = 0; j < L_4; j++) {
        for (int i = 0; i < L_3; i++) {
            o_w_l4[i][j] = w_l4[i][j];
        }
        o_b_l4[j] = b_l4[j];
    }

    for (int j = 0; j < L_5; j++) {
        for (int i = 0; i < L_4; i++) {
            o_w_l5[i][j] = w_l5[i][j];
        }
        o_b_l5[j] = b_l5[j];
    }

    for (int j = 0; j < L_6; j++) {
        for (int i = 0; i < L_5; i++) {
            o_w_l6[i][j] = w_l6[i][j];
        }
        o_b_l6[j] = b_l6[j];
    }
}

template<int N_OUT>
void windower(hls::stream<interface_t> &data, interface_t window[N_OUT]) {
    #pragma HLS PIPELINE II=1 enable_flush
    static interface_t data_window [N_OUT];
    #pragma HLS ARRAY_PARTITION variable=data_window complete dim=0

    for (int i = 0; i < N_OUT-1; i++) {
        #pragma HLS UNROLL
        data_window[i] = data_window[i+1];
    }
    //data_window[N_OUT-1] = short2fxd(data.read());
    data_window[N_OUT-1] = data.read();

    for (int i = 0 ; i < N_OUT; i++) {
        #pragma HLS UNROLL
        window[i] = data_window[i];
    }
}

template<class data_T, class res_T, class weight_T, class bias_T, class acc_T, class ref_T, int N_IN, int N_OUT, int N_REF>
void compute_layer(
    data_T    data[N_IN],
    res_T     res[N_OUT],
    weight_T  weights[N_IN][N_OUT],
    bias_T    biases[N_OUT],
    ref_T    in[N_REF],
    ref_T    out[N_REF]) {
    #pragma HLS PIPELINE II=1 enable_flush
    acc_T acc[N_OUT];

	#pragma HLS ARRAY_RESHAPE variable=weights complete dim=2
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=1

    Product: for(int jj = 0; jj < N_OUT; jj++) {
        acc[jj] = 0;
        #pragma HLS UNROLL
        NewInput: for(int ii = 0; ii < N_IN; ii++) {
            #pragma HLS UNROLL
            acc[jj] += data[ii] * weights[ii][jj];
        }
    }

    Result: for(int ires = 0; ires < N_OUT; ires++) {
        #pragma HLS UNROLL
        res_T temp;
        temp = (res_T) (acc[ires] + (acc_T) biases[ires]);
        res[ires] = temp;
    }

    for(int i = 0; i < N_REF; i++) {
        #pragma HLS UNROLL
        out[i] = in[i];
    }
}

template<class data_T, class res_T, int N_IN>
void  relu(data_T data[N_IN], res_T res[N_IN]) {
    #pragma HLS PIPELINE II=1 enable_flush
    data_T datareg;
    for (int ii=0; ii<N_IN; ii++) {
        #pragma HLS UNROLL
        datareg = data[ii];
        if (datareg > 0) res[ii] = datareg;
        else res[ii] = 0;
    }
}

template<int N_IN>
void l2norm(interface_t pred[N_IN], interface_t grnd[N_IN], interface_t &result) {
    #pragma HLS PIPELINE II=1 enable_flush
    interface_t res = 0;
    for (int i = 0; i < N_IN; i++) {
        #pragma HLS UNROLL
        interface_t err;
        err = (pred[i] - grnd[i]);
        interface_t accu;
        accu = err*err;
        res += (interface_t) accu;
    }
    result = res;
}


void nn(interface_t in[LAYER_1],
        interface_t out[LAYER_1],
        interface_t ref[LAYER_1],
        weight_t p_w_l1[LAYER_1][LAYER_1],
        weight_t p_w_l2[LAYER_1][LAYER_2],
        weight_t p_w_l3[LAYER_2][LAYER_3],
        weight_t p_w_l4[LAYER_3][LAYER_3],
        weight_t p_w_l5[LAYER_3][LAYER_2],
        weight_t p_w_l6[LAYER_2][LAYER_1],
        bias_t p_b_l1[LAYER_1],
        bias_t p_b_l2[LAYER_2],
        bias_t p_b_l3[LAYER_3],
        bias_t p_b_l4[LAYER_3],
        bias_t p_b_l5[LAYER_2],
        bias_t p_b_l6[LAYER_1]
        ) {
    #pragma HLS PIPELINE II=1 enable_flush

    // ****************************************
    // NETWORK INSTATIATION
    // ****************************************

    // Layer 1
    /*
    layer1_t a_l1[LAYER_1];
    layer1_t h_l1[LAYER_1];
    interface_t r_l1[LAYER_1];
    compute_layer<interface_t, layer1_t, weight_t, bias_t, accum_t, interface_t, LAYER_1, LAYER_1, LAYER_1>(h_l0, a_l1, p_w_l1, p_b_l1, h_l0, r_l1);
    relu<layer1_t, layer1_t, LAYER_1>(a_l1, h_l1);
    */

    // Layer 2
    layer2_t a_l2[LAYER_2];
    layer2_t h_l2[LAYER_2];
    interface_t r_l2[LAYER_1];
    compute_layer<interface_t, layer2_t, weight_t, bias_t, accum_t, interface_t, LAYER_1, LAYER_2, LAYER_1>(in, a_l2, p_w_l2, p_b_l2, in, r_l2);
    relu<layer2_t, layer2_t, LAYER_2>(a_l2, h_l2);

    // Layer 3
    layer3_t a_l3[LAYER_3];
    layer3_t h_l3[LAYER_3];
    interface_t r_l3[LAYER_1];
    compute_layer<layer2_t, layer3_t, weight_t, bias_t, accum_t, interface_t, LAYER_2, LAYER_3, LAYER_1>(h_l2, a_l3, p_w_l3, p_b_l3, r_l2, r_l3);
    relu<layer3_t, layer3_t, LAYER_3>(a_l3, h_l3);

    // Layer 4
    /*
    layer4_t a_l4[LAYER_3];
    layer4_t h_l4[LAYER_3];
    interface_t r_l4[LAYER_1];
    compute_layer<layer3_t, layer4_t, weight_t, bias_t, accum_t, interface_t, LAYER_3, LAYER_3, LAYER_1>(h_l3, a_l4, p_w_l4, p_b_l4, r_l3, r_l4);
    relu<layer4_t, layer4_t, LAYER_3>(a_l4, h_l4);
    */

    // Layer 5
    layer5_t a_l5[LAYER_2];
    layer5_t h_l5[LAYER_2];
    interface_t r_l5[LAYER_1];
    compute_layer<layer3_t, layer5_t, weight_t, bias_t, accum_t, interface_t, LAYER_3, LAYER_2, LAYER_1>(h_l3, a_l5, p_w_l5, p_b_l5, r_l3, r_l5);
    relu<layer5_t, layer5_t, LAYER_2>(a_l5, h_l5);

    // Layer 6
    interface_t h_l6[LAYER_1];
    interface_t r_l6[LAYER_1];
    compute_layer<layer5_t, interface_t, weight_t, bias_t, accum_t, interface_t, LAYER_2, LAYER_1, LAYER_1>(h_l5, out, p_w_l6, p_b_l6, r_l5, ref);

}

// AXI-Stream port type is compatible with pointer, reference, & array input / ouputs only
// See UG902 Vivado High Level Synthesis guide (2014.4) pg 157 Figure 1-49
void auto_enc(
      hls::stream<interface_t> &data,
      hls::stream<interface_t> &res,
      unsigned short &const_size_in,
      unsigned short &const_size_out,
      unsigned int &rd_addr,
      unsigned int &wr_addr,
      interface_t &wr_val,
      interface_t &rd_val
      )
{
    // Remove ap ctrl ports (ap_start, ap_ready, ap_idle, etc) since we only use the AXI-Stream ports
    #pragma HLS INTERFACE ap_ctrl_none port=return

    #pragma HLS DATAFLOW

    // Connect size indicators
    #pragma HLS INTERFACE ap_none port=const_size_in
    #pragma HLS INTERFACE ap_none port=const_size_out
    const_size_in   = LAYER_1;
    const_size_out  = LAYER_1;

    // Weight and Biases Management
    // Address:
    //  * 31      = Weight (1) or Bias (0)
    //  * 30 - 27 = Layer Number
    //  * 26 - 16  = Weight Location 1
    //  * 15 - 0  = Weight Location 2
    #pragma HLS INTERFACE ap_none port=rd_addr
    #pragma HLS INTERFACE ap_none port=wr_addr
    #pragma HLS INTERFACE ap_none port=wr_val
    #pragma HLS INTERFACE ap_none port=rd_val

    // ****************************************
    // MEMORY INTERFACE
    // ****************************************
    weight_t p_w_l1[LAYER_1][LAYER_1];
    weight_t p_w_l2[LAYER_1][LAYER_2];
    weight_t p_w_l3[LAYER_2][LAYER_3];
    weight_t p_w_l4[LAYER_3][LAYER_3];
    weight_t p_w_l5[LAYER_3][LAYER_2];
    weight_t p_w_l6[LAYER_2][LAYER_1];
    bias_t p_b_l1[LAYER_1];
    bias_t p_b_l2[LAYER_2];
    bias_t p_b_l3[LAYER_3];
    bias_t p_b_l4[LAYER_3];
    bias_t p_b_l5[LAYER_2];
    bias_t p_b_l6[LAYER_1];

    weight_server<LAYER_1, LAYER_2, LAYER_3, LAYER_3, LAYER_2, LAYER_1>(wr_addr, wr_val, rd_addr, rd_val, p_w_l1, p_w_l2, p_w_l3, p_w_l4, p_w_l5, p_w_l6, p_b_l1, p_b_l2, p_b_l3, p_b_l4, p_b_l5, p_b_l6);

    // Input Conversion
    interface_t in[LAYER_1];
    interface_t out[LAYER_1];
    interface_t ref[LAYER_1];
    windower<LAYER_1>(data, in);

    nn(in, out, ref, p_w_l1, p_w_l2, p_w_l3, p_w_l4, p_w_l5, p_w_l6, p_b_l1, p_b_l2, p_b_l3, p_b_l4, p_b_l5, p_b_l6);

    interface_t l2norm_res;
    l2norm<LAYER_1>(out, ref, l2norm_res);

    // Output Conversion
    res << l2norm_res;
}
