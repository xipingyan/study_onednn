// Refer: 3rd/oneDNN/examples/primitives/convolution.cpp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_types.h"
#include "my_test.hpp"
#include "my_log.hpp"

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

void get_param_from_macro(bool &enforce_bf16, int &N, int &IC, int &OC, int& LOOP_NUM)
{
    std::cout << "==========================\n";
    std::cout << "ENV: \n";
    std::cout << " -ONEDNN_VERBOSE (Check oneDNN log): $ export ONEDNN_VERBOSE=2 \n";
    std::cout << " -enforce_bf16 (default 0): $ export enforce_bf16=1 \n";
    std::cout << " -N (batch size, default 3): $ export N=1 \n";
    std::cout << " -IC (input channels, default 32): $ export IC=32 \n";
    std::cout << " -OC (output channels, default 64): $ export OC=64 \n";
    std::cout << " -LOOP_NUM (loop count, default 10000): $ export LOOP_NUM=10000 \n";

    std::cout << "Other param: \n"
              << " -IH = 13, // input height\n"
              << " -IW = 13, // input width\n"
              << " -KH = 3, // weights height\n"
              << " -KW = 3, // weights width\n"
              << " -PH_L = 1, // height padding: left\n"
              << " -PH_R = 1, // height padding: right\n"
              << " -PW_L = 1, // width padding: left\n"
              << " -PW_R = 1, // width padding: right\n"
              << " -SH = 4, // height-wise stride\n"
              << " -SW = 4, // width-wise stride \n";
    std::cout << "==========================\n";

    if (std::getenv("enforce_bf16"))
    {
        enforce_bf16 = static_cast<bool>(std::atoi(std::getenv("enforce_bf16")));
    }

#define GET_ENV(ENV_NAME, RET_VAL)                                              \
    if (std::getenv(ENV_NAME))                                                  \
    {                                                                           \
        RET_VAL = std::atoi(std::getenv(ENV_NAME));                             \
        std::cout << "From ENV: " << ENV_NAME << " = " << RET_VAL << std::endl; \
    }                                                                           \
    else                                                                        \
    {                                                                           \
        std::cout << "Default: " << ENV_NAME << " = " << RET_VAL << std::endl;  \
    }

    GET_ENV("N", N);
    GET_ENV("IC", IC);
    GET_ENV("OC", OC);
    GET_ENV("LOOP_NUM", LOOP_NUM);
}

void convolution_example(dnnl::engine::kind engine_kind)
{
    bool enforce_bf16 = false;
    int N = 3;   // batch size
    int IC = 32; // input channels
    int OC = 64; // output channels
    int LOOP_NUM = 10000;
    get_param_from_macro(enforce_bf16, N, IC, OC, LOOP_NUM);
    auto GetDT = [&]()
    {
        return enforce_bf16 ? dt::bf16 : dt::f32;
    };

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim
        IH = 13,                               // input height
        IW = 13,                               // input width
        KH = 3,                                // weights height
        KW = 3,                                // weights width
        PH_L = 1,                              // height padding: left
        PH_R = 1,                              // height padding: right
        PW_L = 1,                              // width padding: left
        PW_R = 1,                              // width padding: right
        SH = 4,                                // height-wise stride
        SW = 4,                                // width-wise stride
        OH = (IH - KH + PH_L + PH_R) / SH + 1, // output height
        OW = (IW - KW + PW_L + PW_R) / SW + 1; // output width

    // Source (src), weights, bias, and destination (dst) tensors
    // dimensions.
    memory::dims src_dims = {N, IC, IH, IW};
    memory::dims weights_dims = {OC, IC, KH, KW};
    memory::dims bias_dims = {OC};
    memory::dims dst_dims = {N, OC, OH, OW};

    // Strides, padding dimensions.
    memory::dims strides_dims = {SH, SW};
    memory::dims padding_dims_l = {PH_L, PW_L};
    memory::dims padding_dims_r = {PH_R, PW_R};

    // Allocate buffers.
    std::vector<float> src_data(product(src_dims));
    std::vector<float> weights_data(product(weights_dims));
    std::vector<float> bias_data(OC);
    std::vector<float> dst_data(product(dst_dims));

    // Initialize src, weights, and dst tensors.
    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });
    std::generate(weights_data.begin(), weights_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 2.f);
    });
    std::generate(bias_data.begin(), bias_data.end(), []() {
        static int i = 0;
        return std::tanh(float(i++));
    });

    // Create memory objects for tensor data (src, weights, dst). In this
    // example, NCHW layout is assumed for src and dst, and OIHW for weights.
    auto user_src_mem = memory({src_dims, GetDT(), tag::nchw}, engine);
    auto user_weights_mem = memory({weights_dims, GetDT(), tag::oihw}, engine);
    auto user_dst_mem = memory({dst_dims, GetDT(), tag::nchw}, engine);

    // Create memory descriptors with format_tag::any for the primitive. This
    // enables the convolution primitive to choose memory layouts for an
    // optimized primitive implementation, and these layouts may differ from the
    // ones provided by the user.
    auto conv_src_md = memory::desc(src_dims, GetDT(), tag::any);
    auto conv_weights_md = memory::desc(weights_dims, GetDT(), tag::any);
    auto conv_dst_md = memory::desc(dst_dims, GetDT(), tag::any);

    // Create memory descriptor and memory object for input bias.
    auto user_bias_md = memory::desc(bias_dims, GetDT(), tag::a);
    auto user_bias_mem = memory(user_bias_md, engine);

    // Write data to memory object's handle.
    write_to_dnnl_memory(src_data.data(), user_src_mem);
    write_to_dnnl_memory(weights_data.data(), user_weights_mem);
    write_to_dnnl_memory(bias_data.data(), user_bias_mem);

    // Create primitive post-ops (ReLU).
    const float alpha = 0.f;
    const float beta = 0.f;
    post_ops conv_ops;
    conv_ops.append_eltwise(algorithm::eltwise_relu, alpha, beta);
    primitive_attr conv_attr;
    conv_attr.set_post_ops(conv_ops);

    // Create primitive descriptor.
    auto conv_pd = convolution_forward::primitive_desc(engine,
            prop_kind::forward_training, algorithm::convolution_direct,
            conv_src_md, conv_weights_md, user_bias_md, conv_dst_md,
            strides_dims, padding_dims_l, padding_dims_r, conv_attr);

    // For now, assume that the src, weights, and dst memory layouts generated
    // by the primitive and the ones provided by the user are identical.
    auto conv_src_mem = user_src_mem;
    auto conv_weights_mem = user_weights_mem;
    auto conv_dst_mem = user_dst_mem;

    // Reorder the data in case the src and weights memory layouts generated by
    // the primitive and the ones provided by the user are different. In this
    // case, we create additional memory objects with internal buffers that will
    // contain the reordered data. The data in dst will be reordered after the
    // convolution computation has finalized.
    if (conv_pd.src_desc() != user_src_mem.get_desc()) {
        conv_src_mem = memory(conv_pd.src_desc(), engine);
        reorder(user_src_mem, conv_src_mem)
                .execute(engine_stream, user_src_mem, conv_src_mem);
    }

    if (conv_pd.weights_desc() != user_weights_mem.get_desc()) {
        conv_weights_mem = memory(conv_pd.weights_desc(), engine);
        reorder(user_weights_mem, conv_weights_mem)
                .execute(engine_stream, user_weights_mem, conv_weights_mem);
    }

    if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
        conv_dst_mem = memory(conv_pd.dst_desc(), engine);
    }

    // Create the primitive.
    auto conv_prim = convolution_forward(conv_pd);

    // Primitive arguments.
    std::unordered_map<int, memory> conv_args;
    conv_args.insert({DNNL_ARG_SRC, conv_src_mem});
    conv_args.insert({DNNL_ARG_WEIGHTS, conv_weights_mem});
    conv_args.insert({DNNL_ARG_BIAS, user_bias_mem});
    conv_args.insert({DNNL_ARG_DST, conv_dst_mem});

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < LOOP_NUM; i++)
    {
        // Primitive execution: convolution with ReLU.
        conv_prim.execute(engine_stream, conv_args);

        // Reorder the data in case the dst memory descriptor generated by the
        // primitive and the one provided by the user are different.
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Ran time = " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;

    if (conv_pd.dst_desc() != user_dst_mem.get_desc())
    {
        reorder(conv_dst_mem, user_dst_mem)
            .execute(engine_stream, conv_dst_mem, user_dst_mem);
    }
    else
        user_dst_mem = conv_dst_mem;

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(dst_data.data(), user_dst_mem);
}

void convolution_3d_example(dnnl::engine::kind engine_kind)
{
    // mb1440_ic3id2ih14iw14_oc1280od1oh1ow1_kd2kh14kw14sd2sh14sw14pd1ph1pw1
    bool enforce_f16 = engine_kind == dnnl::engine::kind::gpu ? true : false;
    int N = 1440;   // batch size
    int IC = 3; // input channels
    int OC = 1280; // output channels
    int LOOP_NUM = 100;
    // get_param_from_macro(enforce_bf16, N, IC, OC, LOOP_NUM);
    auto GetDT = [&]()
    {
        return enforce_f16 ? dt::f16 : dt::f32;
    };

    auto dev_num = dnnl::engine::get_count(engine_kind);
    std::cout << "== Device [" << (int)engine_kind << "] count: " << dev_num << std::endl;

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim
        ID = 2,                                // input depth
        IH = 14,                               // input height
        IW = 14,                               // input width
        KD = 2,                                // weights depth
        KH = 14,                               // weights height
        KW = 14,                               // weights width
        PD = 0,
        PH_L = 0,                              // height padding: left
        PH_R = 0,                              // height padding: right
        PW_L = 0,                              // width padding: left
        PW_R = 0,                              // width padding: right
        SD = 2,                                // height-wise stride
        SH = 14,                                // height-wise stride
        SW = 14,                                // width-wise stride
        OD = (ID - KD + PW_L + PW_R) / SW + 1, // output depth
        OH = (IH - KH + PH_L + PH_R) / SH + 1, // output height
        OW = (IW - KW + PW_L + PW_R) / SW + 1; // output width

    // Source (src), weights, bias, and destination (dst) tensors
    // dimensions.
    memory::dims src_dims = {N, IC, ID, IH, IW};
    memory::dims weights_dims = {OC, IC, KD, KH, KW};
    memory::dims bias_dims = {OC};
    memory::dims dst_dims = {N, OC, OD, OH, OW};
    auto print_dim = [](memory::dims dms, std::string prefix) {
        std::cout << prefix << " = [";
        for(auto i = 0 ; i < dms.size(); i++) {
            std::cout << dms[i] << ", ";
        }
        std::cout << "]" << std::endl;
    };
    print_dim(src_dims, "src_dims");
    print_dim(weights_dims, "weights_dims");
    print_dim(bias_dims, "bias_dims");
    print_dim(dst_dims, "dst_dims");    

    // Strides, padding dimensions.
    memory::dims strides_dims = {SD, SH, SW};
    memory::dims padding_dims_l = {PD, PH_L, PW_L};
    memory::dims padding_dims_r = {PD, PH_R, PW_R};

    // Allocate buffers.
    std::vector<float> src_data(product(src_dims));
    std::vector<float> weights_data(product(weights_dims));
    std::vector<float> bias_data(OC);
    std::vector<float> dst_data(product(dst_dims));

    // Initialize src, weights, and dst tensors.
    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });
    std::generate(weights_data.begin(), weights_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 2.f);
    });
    std::generate(bias_data.begin(), bias_data.end(), []() {
        static int i = 0;
        return std::tanh(float(i++));
    });

    // Create memory objects for tensor data (src, weights, dst). In this
    // example, NCHW layout is assumed for src and dst, and OIHW for weights.
    auto user_src_mem = memory({src_dims, GetDT(), tag::ncdhw}, engine);
    auto user_weights_mem = memory({weights_dims, GetDT(), tag::oidhw}, engine);
    auto user_dst_mem = memory({dst_dims, GetDT(), tag::ncdhw}, engine);

    // Create memory descriptors with format_tag::any for the primitive. This
    // enables the convolution primitive to choose memory layouts for an
    // optimized primitive implementation, and these layouts may differ from the
    // ones provided by the user.
    auto conv_src_md = memory::desc(src_dims, GetDT(), tag::any);
    auto conv_weights_md = memory::desc(weights_dims, GetDT(), tag::any);
    auto conv_dst_md = memory::desc(dst_dims, GetDT(), tag::any);

    // Create memory descriptor and memory object for input bias.
    auto user_bias_md = memory::desc(bias_dims, GetDT(), tag::a);
    auto user_bias_mem = memory(user_bias_md, engine);

    // Write data to memory object's handle.
    write_to_dnnl_memory(src_data.data(), user_src_mem);
    write_to_dnnl_memory(weights_data.data(), user_weights_mem);
    write_to_dnnl_memory(bias_data.data(), user_bias_mem);

    // Create primitive post-ops (ReLU).
    const float alpha = 0.f;
    const float beta = 0.f;
    // post_ops conv_ops;
    // conv_ops.append_eltwise(algorithm::eltwise_relu, alpha, beta);
    // primitive_attr conv_attr;
    // conv_attr.set_post_ops(conv_ops);

    // Create primitive descriptor.
    auto conv_pd = convolution_forward::primitive_desc(engine,
            prop_kind::forward, algorithm::convolution_direct,
            conv_src_md, conv_weights_md, user_bias_md, conv_dst_md,
            strides_dims, padding_dims_l, padding_dims_r); //conv_attr

    // For now, assume that the src, weights, and dst memory layouts generated
    // by the primitive and the ones provided by the user are identical.
    auto conv_src_mem = user_src_mem;
    auto conv_weights_mem = user_weights_mem;
    auto conv_dst_mem = user_dst_mem;

    // Reorder the data in case the src and weights memory layouts generated by
    // the primitive and the ones provided by the user are different. In this
    // case, we create additional memory objects with internal buffers that will
    // contain the reordered data. The data in dst will be reordered after the
    // convolution computation has finalized.
    if (conv_pd.src_desc() != user_src_mem.get_desc()) {
        conv_src_mem = memory(conv_pd.src_desc(), engine);
        reorder(user_src_mem, conv_src_mem)
                .execute(engine_stream, user_src_mem, conv_src_mem);
    }

    if (conv_pd.weights_desc() != user_weights_mem.get_desc()) {
        conv_weights_mem = memory(conv_pd.weights_desc(), engine);
        reorder(user_weights_mem, conv_weights_mem)
                .execute(engine_stream, user_weights_mem, conv_weights_mem);
    }

    if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
        conv_dst_mem = memory(conv_pd.dst_desc(), engine);
    }

    // Create the primitive.
    auto conv_prim = convolution_forward(conv_pd);

    // Primitive arguments.
    std::unordered_map<int, memory> conv_args;
    conv_args.insert({DNNL_ARG_SRC, conv_src_mem});
    conv_args.insert({DNNL_ARG_WEIGHTS, conv_weights_mem});
    conv_args.insert({DNNL_ARG_BIAS, user_bias_mem});
    conv_args.insert({DNNL_ARG_DST, conv_dst_mem});

    auto t1 = std::chrono::high_resolution_clock::now();
    conv_prim.execute(engine_stream, conv_args);
    engine_stream.wait();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "== warmup time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;

    for (int i = 0; i < LOOP_NUM; i++)
    {
        t1 = std::chrono::high_resolution_clock::now();
        conv_prim.execute(engine_stream, conv_args);
        engine_stream.wait();   
        t2 = std::chrono::high_resolution_clock::now();
        std::cout << "== Ran [" << i << "] time = " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " micro sec" << std::endl;
    }
 
    if (conv_pd.dst_desc() != user_dst_mem.get_desc())
    {
        reorder(conv_dst_mem, user_dst_mem)
            .execute(engine_stream, conv_dst_mem, user_dst_mem);
    }
    else
        user_dst_mem = conv_dst_mem;

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(dst_data.data(), user_dst_mem);
}

void depthwise_convolution_example(dnnl::engine::kind engine_kind) {

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim N = 3, // batch size
            G = 32, // channel groups
            IC = 32, // input channels
            IH = 13, // input height
            IW = 13, // input width
            OC = 32, // output channels
            KH = 3, // weights height
            KW = 3, // weights width
            PH_L = 1, // height padding: left
            PH_R = 1, // height padding: right
            PW_L = 1, // width padding: left
            PW_R = 1, // width padding: right
            SH = 4, // height-wise stride
            SW = 4, // width-wise stride
            OH = (IH - KH + PH_L + PH_R) / SH + 1, // output height
            OW = (IW - KW + PW_L + PW_R) / SW + 1; // output width

    // Source (src), weights, bias, and destination (dst) tensors dimensions.
    memory::dims src_dims = {N, IC, IH, IW};
    memory::dims weights_dims = {G, OC / G, IC / G, KH, KW};
    memory::dims bias_dims = {OC};
    memory::dims dst_dims = {N, OC, OH, OW};

    // Strides, padding dimensions.
    memory::dims strides_dims = {SH, SW};
    memory::dims padding_dims_l = {PH_L, PW_L};
    memory::dims padding_dims_r = {PH_R, PW_R};

    // Allocate buffers.
    std::vector<float> src_data(product(src_dims));
    std::vector<float> weights_data(product(weights_dims));
    std::vector<float> bias_data(OC);
    std::vector<float> dst_data(product(dst_dims));

    // Initialize src, weights, and dst tensors.
    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });
    std::generate(weights_data.begin(), weights_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 2.f);
    });
    std::generate(bias_data.begin(), bias_data.end(), []() {
        static int i = 0;
        return std::tanh(float(i++));
    });

    // Create memory objects for tensor data (src, weights, dst). In this
    // example, NCHW layout is assumed for src and dst, and OIHW for weights.
    auto user_src_mem = memory({src_dims, dt::f32, tag::nchw}, engine);
    auto user_weights_mem = memory({weights_dims, dt::f32, tag::goihw}, engine);
    auto user_dst_mem = memory({dst_dims, dt::f32, tag::nchw}, engine);

    // Create memory descriptors with format_tag::any for the primitive. This
    // enables the convolution primitive to choose memory layouts for an
    // optimized primitive implementation, and these layouts may differ from the
    // ones provided by the user.
    auto conv_src_md = memory::desc(src_dims, dt::f32, tag::any);
    auto conv_weights_md = memory::desc(weights_dims, dt::f32, tag::any);
    auto conv_dst_md = memory::desc(dst_dims, dt::f32, tag::any);

    // Create memory descriptor and memory object for input bias.
    auto user_bias_md = memory::desc(bias_dims, dt::f32, tag::a);
    auto user_bias_mem = memory(user_bias_md, engine);

    // Write data to memory object's handle.
    write_to_dnnl_memory(src_data.data(), user_src_mem);
    write_to_dnnl_memory(weights_data.data(), user_weights_mem);
    write_to_dnnl_memory(bias_data.data(), user_bias_mem);

    // Create primitive post-ops (ReLU).
    const float alpha = 0.f;
    const float beta = 0.f;
    post_ops conv_ops;
    conv_ops.append_eltwise(algorithm::eltwise_relu, alpha, beta);
    primitive_attr conv_attr;
    conv_attr.set_post_ops(conv_ops);

    // Create primitive descriptor.
    auto conv_pd = convolution_forward::primitive_desc(engine,
            prop_kind::forward_training, algorithm::convolution_direct,
            conv_src_md, conv_weights_md, user_bias_md, conv_dst_md,
            strides_dims, padding_dims_l, padding_dims_r, conv_attr);

    // For now, assume that the src, weights, and dst memory layouts generated
    // by the primitive and the ones provided by the user are identical.
    auto conv_src_mem = user_src_mem;
    auto conv_weights_mem = user_weights_mem;
    auto conv_dst_mem = user_dst_mem;

    // Reorder the data in case the src and weights memory layouts generated by
    // the primitive and the ones provided by the user are different. In this
    // case, we create additional memory objects with internal buffers that will
    // contain the reordered data. The data in dst will be reordered after the
    // convolution computation has finalized.
    if (conv_pd.src_desc() != user_src_mem.get_desc()) {
        conv_src_mem = memory(conv_pd.src_desc(), engine);
        reorder(user_src_mem, conv_src_mem)
                .execute(engine_stream, user_src_mem, conv_src_mem);
    }

    if (conv_pd.weights_desc() != user_weights_mem.get_desc()) {
        conv_weights_mem = memory(conv_pd.weights_desc(), engine);
        reorder(user_weights_mem, conv_weights_mem)
                .execute(engine_stream, user_weights_mem, conv_weights_mem);
    }

    if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
        conv_dst_mem = memory(conv_pd.dst_desc(), engine);
    }

    // Create the primitive.
    auto conv_prim = convolution_forward(conv_pd);

    // Primitive arguments.
    std::unordered_map<int, memory> conv_args;
    conv_args.insert({DNNL_ARG_SRC, conv_src_mem});
    conv_args.insert({DNNL_ARG_WEIGHTS, conv_weights_mem});
    conv_args.insert({DNNL_ARG_BIAS, user_bias_mem});
    conv_args.insert({DNNL_ARG_DST, conv_dst_mem});

    // Primitive execution: convolution with ReLU.
    conv_prim.execute(engine_stream, conv_args);

    // Reorder the data in case the dst memory descriptor generated by the
    // primitive and the one provided by the user are different.
    if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
        reorder(conv_dst_mem, user_dst_mem)
                .execute(engine_stream, conv_dst_mem, user_dst_mem);
    } else
        user_dst_mem = conv_dst_mem;

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(dst_data.data(), user_dst_mem);
}

int getting_started_test_conv(int argc, char **argv) {
    std::cout << "Start: convolution_example ..................." << std::endl;
    int exit_code = 0;
    // exit_code = handle_example_errors(
    //         convolution_example, parse_engine_kind(argc, argv));
    // if (exit_code != 0) {
    //     PERR << "convolution_example FAIL." << std::endl;
    //     return exit_code;
    // }

    std::cout << "Start: convolution_3d_example ..................." << std::endl;
    exit_code = handle_example_errors(
        convolution_3d_example, parse_engine_kind(argc, argv));
    if (exit_code != 0)
    {
        PERR << "convolution_example FAIL." << std::endl;
        return exit_code;
    }

    // exit_code = handle_example_errors(
    //     depthwise_convolution_example, parse_engine_kind(argc, argv));
    // if (exit_code != 0)
    // {
    //     PERR << "depthwise_convolution_example FAIL." << std::endl;
    //     return exit_code;
    // }
    return exit_code;
}
