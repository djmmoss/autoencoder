import numpy as np

def print_weights(data_path, n_weights):
        for w in n_weights:
                weights = n_weights[w].eval()
                shp = weights.shape
                f = open(data_path + str(w) + ".h", "w")
                f.write("static weight_t " + str(w) + "[" + str(shp[0]) + "][" + str(shp[1]) + "] = {{")
                for i in range(shp[0]-1):
                        for j in range(shp[1]-1):
                                f.write(str(weights[i][j]) + ",")
                        f.write(str(weights[i][shp[1]-1]) + "},\n{")
                for j in range(shp[1]-1):
                        f.write(str(weights[shp[0]-1][j]) + ",")
                f.write(str(weights[shp[0]-1][shp[1]-1]) + "}};\n")
                f.flush()
                f.close()

def print_biases(data_path, n_biases):
        for b in n_biases:
                biases = n_biases[b].eval()
                shp = biases.shape
                f = open(data_path + str(b) + ".h", "w")
                f.write("static bias_t " + str(b) + "[" + str(shp[0]) + "] = {")
                for j in range(shp[0]-1):
                        f.write(str(biases[j]) + ",")
                f.write(str(biases[shp[0]-1]) + "};\n")
                f.flush()
                f.close()

def print_network(data_path, L1, L2, L3):
    f = open(data_path + "net.h", "w")
    f.write("#define LAYER_1 " + str(L1) + "\n")
    f.write("#define LAYER_2 " + str(L2) + "\n")
    f.write("#define LAYER_3 " + str(L3) + "\n")
    f.flush()
    f.close()

    f = open(data_path + "noc_block_autoenc_tb.vh", "w")
    f.write("`define TEST_SPP " + str(L1) + "\n")
    f.write("`define TEST_TRL 2\n")
    f.write("`define TEST_ERR 32\n")
    f.flush()
    f.close()

def bitReverse(n):

    assert (n & (n-1) == 0)

    v = np.arange(n).astype(np.int)
    b = np.zeros(n).astype(np.int)
    m = np.log2(n).astype(np.int)

    for k, x in enumerate(v):
        b[k] = np.sum([ 1<<(m-1-i) for i in range(m) if (x>>i)&1 ])

    return b

def print_fft(data_path, n, do_fft=True):

    w_vals = np.arange(0, n)
    pfw = np.exp(-2j*np.pi*(w_vals/n))
    pfws = "static cplx pfw [%d] = {" % (n)
    for w in pfw[:-1]:
        pfws += "{ " + str(np.real(w)) + ", " + str(np.imag(w)) + "},\n"
    pfws += "{ " + str(np.real(pfw[-1])) + ", " + str(np.imag(pfw[-1])) + "}};\n"

    func_call = "void fft(cplx in[%d], cplx out[%d]) {\n" % (n, n)
    top = """
    #pragma HLS PIPELINE II=1 enable_flush
    unsigned int stage, block, j, iw=0;
    unsigned int pa, pb, qa, qb;
    cplx ft1a, ft1b, ft2a, ft2b, ft3a, ft3b;

    cplx pfs[%d];

    for (int i = 0; i < %d; i++) {
        #pragma HLS UNROLL
        pfs[i].re = in[i].re;
        pfs[i].im = in[i].im;
    }
""" % (n, n)

    body = ""
    log_n = int(np.log2(n))
    stride = int(n/2)
    edirts = 1
    for _ in range(log_n-2):
        stage = """
    for( block=0; block<%d; block+=%d*2 ) {
        #pragma HLS UNROLL
        pa = block;
        pb = block + %d/2;
        qa = block + %d;
        qb = block + %d/2 + %d;
        iw = 0;
        for( j=0; j < %d/2; j++ ) { //2bufflies/loop
            #pragma HLS UNROLL
            //add
            ft1a.re = pfs[pa+j].re + pfs[qa+j].re;
            ft1a.im = pfs[pa+j].im + pfs[qa+j].im;
            ft1b.re = pfs[pb+j].re + pfs[qb+j].re;
            ft1b.im = pfs[pb+j].im + pfs[qb+j].im;
            //sub
            ft2a.re = pfs[pa+j].re - pfs[qa+j].re;
            ft2a.im = pfs[pa+j].im - pfs[qa+j].im;
            ft2b.re = pfs[pb+j].re - pfs[qb+j].re;
            ft2b.im = pfs[pb+j].im - pfs[qb+j].im;
            pfs[pa+j].re = ft1a.re; //store adds
            pfs[pa+j].im = ft1a.im; //store adds
            pfs[pb+j].re = ft1b.re;
            pfs[pb+j].im = ft1b.im;
            //cmul
            pfs[qa+j].re = ft2a.re * pfw[iw].re - ft2a.im * pfw[iw].im;
            pfs[qa+j].im = ft2a.re * pfw[iw].im + ft2a.im * pfw[iw].re;
            //twiddled cmul
            pfs[qb+j].re = ft2b.re * pfw[iw].im + ft2b.im * pfw[iw].re;
            pfs[qb+j].im = -ft2b.re * pfw[iw].re + ft2b.im * pfw[iw].im;
            iw += %d;
        }
    }
""" % (n, stride, stride, stride, stride, stride, stride, edirts)
        body += stage
        stride = stride >> 1
        edirts = edirts << 1

    last_stages = """
    //last two stages
    for( j=0; j<%d; j+=4 ) {
        #pragma HLS UNROLL
        //upper two
        ft1a.re = pfs[j ].re + pfs[j+2].re;
        ft1a.im = pfs[j ].im + pfs[j+2].im;
        ft1b.re = pfs[j+1].re + pfs[j+3].re;
        ft1b.im = pfs[j+1].im + pfs[j+3].im;
        ft2a.re = ft1a.re + ft1b.re;
        ft2a.im = ft1a.im + ft1b.im;
        ft2b.re = ft1a.re - ft1b.re;
        ft2b.im = ft1a.im - ft1b.im;
        //lower two
        //notwiddle
        ft3a.re = pfs[j].re - pfs[j+2].re;
        ft3a.im = pfs[j].im - pfs[j+2].im;
        //twiddle
        ft3b.re = pfs[j+1].im - pfs[j+3].im;
        ft3b.im = -pfs[j+1].re + pfs[j+3].re;
        //store
        pfs[j ].re = ft2a.re;
        pfs[j ].im = ft2a.im;
        pfs[j+1].re = ft2b.re;
        pfs[j+1].im = ft2b.im;
        pfs[j+2].re = ft3a.re + ft3b.re;
        pfs[j+2].im = ft3a.im + ft3b.im;
        pfs[j+3].re = ft3a.re - ft3b.re;
        pfs[j+3].im = ft3a.im - ft3b.im;
    }
""" % (n)

    body += last_stages

    idx = bitReverse(n)

    bit_reverse = ""
    for i in range(n):
        bit = """
    out[%d].re = pfs[%d].re;
    out[%d].im = pfs[%d].im;
""" % (idx[i], i, idx[i], i)
        bit_reverse += bit

    body += bit_reverse

    finish = "}\n"

    if not do_fft:
        body = """
    for (int i = 0; i < %d; i++) {
        #pragma HLS UNROLL
        out[i].re = pfs[i].re;
        out[i].im = pfs[i].im;
    }
""" % (n)


    f = open(data_path + "fft.h", "w")
    f.write(pfws)
    f.write(func_call)
    f.write(top)
    f.write(body)
    f.write(finish)
    f.flush()
    f.close()
