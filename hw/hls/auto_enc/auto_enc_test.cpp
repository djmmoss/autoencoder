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
#include "nnet_helpers.h"
#include "data/net.h"

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
			   	       data[ii][jj] = float2short(newval);
			   	     } else {
			   	       return -2;
			   	     }
		   }
		   if (fscanf(fp, "%f\n", &newval) != 0){
			   data[ii][window-1] = float2short(newval);
		   } else {
			   return -2;
		   }
	   }
	   fclose(fp);
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

  short  result[10][LAYER_1];
  short  data[10][LAYER_1];
  short expected[10][LAYER_1];

  // Load data from file
  int rval = 0;
  rval = read_file<short, 10, LAYER_1>("data/data.out", data);
  rval = read_file<short, 10, LAYER_1>("data/expected.out", expected);
  rval = write_binary_file<short, 10, LAYER_1>("data/test_in.bin", data);

  // Run the basic neural net block
  unsigned short size_in, size_out;
  int err_cnt = 0;

  for (int isample=0; isample < 10; isample++) {
	  hls::stream<short> data_str;
  	  for (int idat=0; idat < LAYER_1; idat++) {
  		  data_str << data[isample][idat];
  	  }

  	  hls::stream<short> res_str;
  	  auto_enc(data_str, res_str, size_in, size_out);

  	  // Print result vector
  	  float err;
      short curr_data;
  	  for (int ii = 0; ii < LAYER_1; ii++) {
  		  curr_data = res_str.read();
          result[isample][ii] = curr_data;
  		  err = short2float(curr_data, I_WIDTH) - short2float(expected[isample][ii], 1);
  		  std::cout << " Expected: " << short2float(expected[isample][ii], 1) << "   Received: " << short2float(curr_data, I_WIDTH) << "  ErrVal: " << err << std::endl;
  		  if (abs(err) > 0.5) err_cnt++;
  	  }
  }

  rval = write_binary_file<short, 10, LAYER_1>("data/test_out.bin", result);
  return err_cnt;
}

