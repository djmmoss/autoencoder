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
			   	       data[ii][jj] = newval;
			   	     } else {
			   	       return -2;
			   	     }
		   }
		   if (fscanf(fp, "%f\n", &newval) != 0){
			   data[ii][window-1] = newval;
		   } else {
			   return -2;
		   }
	   }
	   fclose(fp);
	   return 0;
}


int main(int argc, char **argv)
{
  // DATA FROM UDACITY TENSORFLOW CLASS
  // 1-Layer test

  input_t  data[10][32];
  float answer[10][32];

  // Load data from file
  int rval = 0;
  rval = read_file<input_t, 10, 32>("data/data.out", data);
  rval = read_file<float, 10, 32>("data/expected.out", answer);

  // Run the basic neural net block
  unsigned short size_in, size_out;
  int err_cnt = 0;

  for (int isample=0; isample < 10; isample++) {
	  hls::stream<input_t> data_str;
  	  for (int idat=0; idat < 32; idat++) {
  		  data_str << data[isample][idat];
  	  }

  	  hls::stream<result_t> res_str;
  	  auto_enc(data_str, res_str, size_in, size_out);

  	  // Print result vector
  	  float err, curr_data;
  	  for (int ii = 0; ii < 32; ii++) {
  		  curr_data = res_str.read();
  		  err = curr_data-answer[isample][ii];
  		  std::cout << " Expected: " << answer[isample][ii] << "   Received: " << curr_data << "  ErrVal: " << err << std::endl;
  		  if (abs(err) > 0.5) err_cnt++;
  	  }
  }
  std::cout<< err_cnt << std::endl;
  return err_cnt;
}

