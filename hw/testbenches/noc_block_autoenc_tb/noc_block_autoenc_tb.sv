`timescale 1ns/1ps
`define NS_PER_TICK 1
`define NUM_TEST_CASES 5

`include "sim_exec_report.vh"
`include "sim_clks_rsts.vh"
`include "sim_rfnoc_lib.svh"
`include "noc_block_autoenc_tb.vh"

module noc_block_autoenc_tb();
  `TEST_BENCH_INIT("noc_block_autoenc",`NUM_TEST_CASES,`NS_PER_TICK);
  localparam BUS_CLK_PERIOD = $ceil(1e9/166.67e6);
  localparam CE_CLK_PERIOD  = $ceil(1e9/200e6);
  localparam NUM_CE         = 1;  // Number of Computation Engines / User RFNoC blocks to simulate
  localparam NUM_STREAMS    = 1;  // Number of test bench streams
  `RFNOC_SIM_INIT(NUM_CE, NUM_STREAMS, BUS_CLK_PERIOD, CE_CLK_PERIOD);
  `RFNOC_ADD_BLOCK(noc_block_autoenc, 0);

  localparam SPP = `TEST_SPP;
  localparam TRL = `TEST_TRL;
  localparam ERR = `TEST_ERR;

  /********************************************************
  ** Verification
  ********************************************************/
  initial begin : tb_main
    string s;
    logic [31:0] random_word;
    logic [63:0] readback;
    integer data_file; // file handler
    integer scan_file; // file handler
    integer data_file_ref;

    /********************************************************
    ** Test 1 -- Reset
    ********************************************************/
    `TEST_CASE_START("Wait for Reset");
    while (bus_rst) @(posedge bus_clk);
    while (ce_rst) @(posedge ce_clk);
    `TEST_CASE_DONE(~bus_rst & ~ce_rst);

    /********************************************************
    ** Test 2 -- Check for correct NoC IDs
    ********************************************************/
    `TEST_CASE_START("Check NoC ID");
    // Read NOC IDs
    tb_streamer.read_reg(sid_noc_block_autoenc, RB_NOC_ID, readback);
    $display("Read AUTOENC NOC ID: %16x", readback);
    `ASSERT_ERROR(readback == noc_block_autoenc.NOC_ID, "Incorrect NOC ID");
    `TEST_CASE_DONE(1);

    /********************************************************
    ** Test 3 -- Connect RFNoC blocks
    ********************************************************/
    `TEST_CASE_START("Connect RFNoC blocks");
    `RFNOC_CONNECT(noc_block_tb,noc_block_autoenc,S16,SPP);
    `RFNOC_CONNECT(noc_block_autoenc,noc_block_tb,S16,SPP);
    `TEST_CASE_DONE(1);

    /********************************************************
    ** Test 4 -- Write / readback user registers
    ********************************************************/
    `TEST_CASE_START("Write / readback user registers");
    tb_streamer.write_user_reg(sid_noc_block_autoenc, noc_block_autoenc.SR_SIZE_INPUT, SPP*4);
    tb_streamer.read_user_reg(sid_noc_block_autoenc, noc_block_autoenc.RB_SIZE_INPUT, readback);
    $sformat(s, "User register 0 incorrect readback! Expected: %0d, Actual %0d", readback[31:0], SPP*4);
    `ASSERT_ERROR(readback[31:0] == SPP*4, s);
    random_word = $random();
    tb_streamer.write_user_reg(sid_noc_block_autoenc, noc_block_autoenc.SR_SIZE_OUTPUT, 10*4);
    tb_streamer.read_user_reg(sid_noc_block_autoenc, noc_block_autoenc.RB_SIZE_OUTPUT, readback);
    $sformat(s, "User register 1 incorrect readback! Expected: %0d, Actual %0d", readback[31:0], 10*4);
    `ASSERT_ERROR(readback[31:0] == 10*4, s);
    `TEST_CASE_DONE(1);

    /********************************************************
    ** Test 5 -- Test sequence
    ********************************************************/
    
    `TEST_CASE_START("Test Neural Net Data");
    // Run the test twice to make sure we can recreate results
    
    fork
      begin
        real data_float;
        integer data_int;
        logic [15:0] data_logic;
	logic [7:0] part1, part2;
        data_file = $fopen("test_in.bin", "rb");
        `ASSERT_FATAL(data_file != 0, "Data file could not be opened");
        if (data_file == 0) begin
          $display("data_file handle was NULL");
          $finish;
        end
        $display("Send data from text file");
        for (int i = 0; i < SPP*TRL; i++) begin
          scan_file = $fread(part2, data_file);
          scan_file = $fread(part1, data_file);
	  data_logic = {part1, part2};
          if ( i == (SPP*TRL - 1))
            tb_streamer.push_word({data_logic}, 1 );
          else
            tb_streamer.push_word({data_logic}, 0 );
          $sformat(s, "Pushing word: %f, %d", data_float, data_int);
          //$display(s);
        end
        $fclose(data_file);
      end
      begin
        logic last, p_fail, n_fail;
        logic [15:0] res_logic;
	logic [15:0] ref_logic;
	logic [7:0] part1, part2;
        data_file_ref = $fopen("test_out.bin", "rb");
        `ASSERT_FATAL(data_file_ref != 0, "Output data file could not be opened");
        for (int ii = 0; ii < SPP*TRL; ii++) begin
          tb_streamer.pull_word({res_logic}, last);
          scan_file = $fread(part2, data_file_ref);
          scan_file = $fread(part1, data_file_ref);
	  ref_logic = {part1, part2};
          p_fail = ($signed(ref_logic) - $signed(res_logic)) > ERR;
          n_fail = ($signed(ref_logic) - $signed(res_logic)) < -ERR;
          $sformat(s, "Received Value: %h, Exp: %h, Err: %h, PF: %b, NF: %b", res_logic, ref_logic, ref_logic - res_logic, p_fail , n_fail);
          `ASSERT_ERROR(~n_fail && ~p_fail, s);
        end
        $fclose(data_file_ref);
      end
    join
    `TEST_CASE_DONE(1);


    `TEST_BENCH_DONE;

  end
endmodule
