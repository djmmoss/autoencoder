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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "auto_enc.h"
#include "data/net.h"

#define IN_WIDTH 4

template <class dataType, unsigned int samples, unsigned int window>
int read_file(const char * filename, dataType data[samples][window]) {
	  FILE *fp;
	  fp = fopen(filename, "r");
	  if (fp == 0) {
	    return -1;
	  }
	  // Read data from file
	   float newval;
	   for (int ii = 0; ii < samples; ii++){
		   for (int jj = 0; jj < window-1; jj++) {
			   if (fscanf(fp, "%f,", &newval) != 0){
			   	       data[ii][jj] = float2short(newval, IN_WIDTH);
			   	     } else {
			   	       return -2;
			   	     }
		   }
		   if (fscanf(fp, "%f\n", &newval) != 0){
			   data[ii][window-1] = float2short(newval, IN_WIDTH);
		   } else {
			   return -2;
		   }
	   }
	   fclose(fp);
	   return 0;
}

template <class dataType, unsigned int samples>
int read_file_1D(const char * filename, dataType data[samples]) {
	  FILE *fp;
	  fp = fopen(filename, "r");
	  if (fp == 0) {
	    return -1;
	  }
	  // Read data from file
	   float newval;
	   for (int ii = 0; ii < samples; ii++){
		   if (fscanf(fp, "%f\n", &newval) != 0){
			   //data[ii] = float2short(newval, IN_WIDTH-1);
			   data[ii] = (dataType) newval;
		   } else {
			   return -2;
		   }
	   }
	   fclose(fp);
	   return 0;
}
template <class dataType, unsigned int samples>
int read_file_1Df(const char * filename, dataType data[samples]) {
	  FILE *fp;
	  fp = fopen(filename, "r");
	  if (fp == 0) {
	    return -1;
	  }
	  // Read data from file
	   float newval;
	   for (int ii = 0; ii < samples; ii++){
		   if (fscanf(fp, "%f\n", &newval) != 0){
			   data[ii] = newval;
		   } else {
			   return -2;
		   }
	   }
	   fclose(fp);
	   return 0;
}

template <class dataType, unsigned int samples>
int read_file_1D_cplx(const char * filename, const char * filename2, dataType data[samples]) {
	  FILE *fp, *fp2;
	  fp = fopen(filename, "r");
	  fp2 = fopen(filename2, "r");
	  if (fp == 0) {
	    return -1;
	  }
	  if (fp2 == 0) {
	    return -1;
	  }
	  // Read data from file
	   float newval;
	   for (int ii = 0; ii < samples; ii++){
		   if (fscanf(fp, "%f\n", &newval) != 0){
			   data[ii].re = newval;
		   } else {
			   return -2;
		   }
	   }
	   fclose(fp);
	   for (int ii = 0; ii < samples; ii++){
		   if (fscanf(fp2, "%f\n", &newval) != 0){
			   data[ii].im = newval;
		   } else {
			   return -2;
		   }
	   }
	   fclose(fp2);
	   return 0;
}

template <class dataType, unsigned int samples, unsigned int window>
int write_binary_file(const char * filename, dataType data[samples][window]) {
	  FILE *fp;
	  fp = fopen(filename, "wb");
	  if (fp == 0) {
	    return -1;
	  }

	  // Write data to file
	  for (int ii = 0; ii < samples; ii++){
		   for (int jj = 0; jj < window-1; jj++) {
			dataType newval = (dataType) data[ii][jj];
			fwrite(&newval, sizeof(short), 1, fp);
		   }
	   }
	   fclose(fp);
	   return 0;
}

template <class dataType, unsigned int samples>
int write_binary_file_1D(const char * filename, dataType data[samples]) {
	  FILE *fp;
	  fp = fopen(filename, "wb");
	  if (fp == 0) {
	    return -1;
	  }

	  // Write data to file
	  for (int ii = 0; ii < samples; ii++){
        dataType newval = (dataType) data[ii];
        fwrite(&newval, sizeof(short), 1, fp);
       }
	   fclose(fp);
	   return 0;
}

template <class dataType, unsigned int samples, unsigned int window>
int read_binary_file(const char * filename, dataType data[samples][window]) {
	  FILE *fp;
	  fp = fopen(filename, "rb");
	  if (fp == 0) {
	    return -1;
	  }
	  // Write data to file
	   for (int ii = 0; ii < samples; ii++){
		   for (int jj = 0; jj < window-1; jj++) {
			dataType newval;
			fread(&newval, sizeof(short), 1, fp);
			data[ii][jj] = newval;
		   }
	   }
	   fclose(fp);
	   return 0;
}

int main(int argc, char **argv)
{
  // DATA FROM UDACITY TENSORFLOW CLASS
  // 1-Layer test

  interface_t  result[100];
  cplx  data[100];
  float expected[100];

  // Load data from file
  int rval = 0;
  rval = read_file_1D_cplx<cplx, 100>("data/data_i.out", "data/data_q.out", data);
  rval = read_file_1Df<float, 100>("data/expected.out", expected);
  //rval = write_binary_file_1D<short, 100>("data/test_in.bin", data);

  // Run the basic neural net blocks
  unsigned short size_in, size_out;
  unsigned int wr_addr, rd_addr;
  interface_t wr_val, rd_val;
  int err_cnt = 0;

  for (int isample=0; isample < 100; isample++) {
	  hls::stream<cplx> data_str;
      data_str << data[isample];

  	  hls::stream<interface_t> res_str;
  	  auto_enc(data_str, res_str, size_in, size_out, wr_addr, rd_addr, wr_val, rd_val);

      interface_t curr_data;
      curr_data = res_str.read();
      result[isample] = curr_data;

  	  // Print result vector
      if (isample > (LAYER_1+1)) {
          float err;
          err = (float) curr_data - expected[isample-LAYER_1+1];
          std::cout << " Expected: " << expected[isample-LAYER_1+1] << "   Received: " << curr_data << "  ErrVal: " << err << std::endl;
          //err = curr_data - expected[isample-LAYER_1+1];
          //std::cout << " Expected: " << expected[isample-LAYER_1+1] << "   Received: " << curr_data << "  ErrVal: " << err << std::endl;
          if (abs(err) > 0.5) err_cnt++;
      }
  }

  //rval = write_binary_file_1D<short, 100>("data/test_out.bin", result);
  return err_cnt;
}

