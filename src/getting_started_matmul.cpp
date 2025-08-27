// Refer: 3rd/oneDNN/examples/primitives/convolution.cpp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_types.h"
#include "my_test.hpp"
#include "my_log.hpp"

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

struct gemm_dims_t {
    memory::dim m, n, k;
};

void fill_random(std::vector<float> &out, bool is_integer) {
    static std::vector<float> random_data_i, random_data_f;
    constexpr size_t nrand = 1037;

    if (random_data_i.empty() || random_data_f.empty()) {
        std::mt19937 generator;
        std::uniform_int_distribution<int> dist_i(-16, 15);
        std::uniform_real_distribution<float> dist_f(-1.0f, 1.0f);

        random_data_i.resize(nrand);
        for (auto &d : random_data_i)
            d = static_cast<float>(dist_i(generator));

        random_data_f.resize(nrand);
        for (auto &d : random_data_f)
            d = dist_f(generator);
    }

    auto &rd = is_integer ? random_data_i : random_data_f;

    for (size_t i = 0; i < out.size(); i += nrand) {
        size_t chunk = std::min(nrand, out.size() - i);
        std::memcpy(&out[i], rd.data(), chunk * sizeof(float));
    }
}

const char *get_type_string(memory::data_type type) {
    const char *type_string = "unknown";

#define TYPE_CASE(T) \
    if (type == memory::data_type::T) type_string = #T;
    TYPE_CASE(f16);
    TYPE_CASE(f32);
    TYPE_CASE(f64);
    TYPE_CASE(bf16);
    TYPE_CASE(s8);
    TYPE_CASE(u8);
#undef TYPE_CASE

    return type_string;
}

void print_test_case(memory::data_type type, gemm_dims_t dims) {
    std::cout << '[' << std::setw(4) << get_type_string(type);
    if (dims.m == dims.n && dims.m == dims.k)
        std::cout << " m = n = k = " << dims.m;
    else
        std::cout << " m = " << dims.m << ", n = " << dims.n
                  << ", k = " << dims.k;
    std::cout << "] " << std::flush;
}

struct MM_Inputs
{
    dnnl::memory a_mem;
    dnnl::memory b_mem;
    dnnl::memory c_mem;

    dnnl::memory zp_mem;
    dnnl::memory scale_mem;

    matmul::primitive_desc matmul_pd;
    dnnl::matmul matmul_prim;

    std::unordered_map<int, memory> matmul_args;
};

std::shared_ptr<MM_Inputs> get_inputs(bool is_integer, memory::data_type &type,
                                      dnnl::engine &engine, dnnl::stream &engine_stream,
                                      memory::dims input_dims, memory::dims weight_dims, memory::dims output_dims)
{
    std::shared_ptr<MM_Inputs> input = std::make_shared<MM_Inputs>();

    // Allocate buffers and random-initialize A/B
    std::vector<float> a_data(product(input_dims));
    std::vector<float> b_data(product(weight_dims));
    std::vector<float> c_data(product(output_dims));

    fill_random(a_data, is_integer);
    fill_random(b_data, is_integer);

    // Create memory descriptors and memory objects for src, weights, bias, and
    // dst.
    // jit:gemm:any,undef,
    // src:f16::blocked:abc::f0 
    // wei:u4::blocked:cab::f0 
    // dst:f16::blocked:abc::f0,
    // attr-scratchpad:user attr-fpmath:f16:true 
    // attr-scales:wei:6:f16:128x1 
    // attr-zero-points:wei:6:u8:128x1 
    // attr-post-ops:binary_add:f16:4,,1x1x2048:1x2048x2048

    auto a_md = memory::desc(input_dims, memory::data_type::f16, memory::format_tag::abc);
    auto b_md = memory::desc(weight_dims, memory::data_type::u4, memory::format_tag::cab);
    auto c_md = memory::desc(output_dims, memory::data_type::f16, memory::format_tag::abc);
    
    int N = output_dims[2];
#define ENALBE_BIAS 0               // work
#define ATTRIBUT_POST_OPS 0         // work
#define ATTRIBUT_POST_OPS_BIN_ADD 0 // Not work.
#define ATTRIBUT_SCALE_ZP 1         // Not work.

#if ENALBE_BIAS
    memory::dims bias_dims = {1, 1, N};
    std::vector<float> bias_data(product(bias_dims));
    std::generate(bias_data.begin(), bias_data.end(), []()
                  {
        static int i = 0;
        return std::tanh(float(i++)); });
    auto bias_md = memory::desc(
        bias_dims, memory::data_type::f32, memory::format_tag::abc);
#endif

    auto a_in_md = memory::desc(input_dims, memory::data_type::f32, memory::format_tag::abc);
    auto b_in_md = memory::desc(weight_dims, memory::data_type::f32, memory::format_tag::abc);
    auto binary_md = memory::desc(output_dims, memory::data_type::f16, memory::format_tag::abc);

    auto a_in_mem = memory(a_in_md, engine);
    auto b_in_mem = memory(b_in_md, engine);
#if ENALBE_BIAS
    auto bias_mem = memory(bias_md, engine);
#endif

    // Write data to memory object's handles.
    write_to_dnnl_memory(a_data.data(), a_in_mem);
    write_to_dnnl_memory(b_data.data(), b_in_mem);
#if ENALBE_BIAS
    write_to_dnnl_memory(bias_data.data(), bias_mem);
#endif

    primitive_attr matmul_attr;
#if ATTRIBUT_POST_OPS
    // Create primitive post-ops (ReLU).
    const float alpha = 0.f;
    const float beta = 0.f;
    post_ops matmul_ops;
    matmul_ops.append_eltwise(algorithm::eltwise_relu, alpha, beta);
    matmul_attr.set_post_ops(matmul_ops);
#endif

#if ATTRIBUT_POST_OPS_BIN_ADD
    dnnl::post_ops po;
    po.append_binary(
        dnnl::algorithm::binary_add,
        dnnl::memory::desc(
            input_dims,
            dnnl::memory::data_type::f16,
            dnnl::memory::format_tag::abcd));
    matmul_attr.set_post_ops(po);
#endif

#if ATTRIBUT_SCALE_ZP
    dnnl::memory::dims zp_dims = {128, 1};
    auto scale_md = memory::desc(zp_dims, memory::data_type::f16, memory::format_tag::ab);
    auto scale_mem = memory(scale_md, engine);    
    auto zp_md = memory::desc(zp_dims, memory::data_type::u8, memory::format_tag::ab);
    auto zp_mem = memory(zp_md, engine);
    
    std::vector<float> scale_data(product(zp_dims));
    std::vector<float> zp_data(product(zp_dims));
    fill_random(scale_data, false);
    fill_random(zp_data, true);

    write_to_dnnl_memory(scale_data.data(), scale_mem);
    write_to_dnnl_memory(zp_data.data(), zp_mem);

    // matmul_attr.set_scales_mask(DNNL_ARG_WEIGHTS, 6);
    // matmul_attr.set_zero_points_mask(DNNL_ARG_WEIGHTS, 6);

    // matmul_attr.set_scales(DNNL_ARG_WEIGHTS, 6, zp_dims, dnnl::memory::data_type::f16);
    matmul_attr.set_zero_points(DNNL_ARG_WEIGHTS, 6, zp_dims, dnnl::memory::data_type::u8);
#endif

    // 设置 FP16 数学模式
    matmul_attr.set_fpmath_mode(fpmath_mode::f16);
    matmul_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    // Create primitive descriptor.
#if ENALBE_BIAS
    input->matmul_pd = matmul::primitive_desc(engine, a_md, b_md, bias_md, c_md, matmul_attr);
#else
    input->matmul_pd = matmul::primitive_desc(engine, a_md, b_md, c_md, matmul_attr);
#endif

    // Repack and convert input data.
    input->a_mem = memory(input->matmul_pd.src_desc(), engine);
    reorder(a_in_mem, input->a_mem).execute(engine_stream, a_in_mem, input->a_mem);

    input->b_mem = memory(input->matmul_pd.weights_desc(), engine);
    reorder(b_in_mem, input->b_mem).execute(engine_stream, b_in_mem, input->b_mem);

    input->c_mem = memory(input->matmul_pd.dst_desc(), engine);

    // Create the primitive.
    input->matmul_prim = matmul(input->matmul_pd);

    input->matmul_args.insert({DNNL_ARG_SRC, input->a_mem});
    input->matmul_args.insert({DNNL_ARG_WEIGHTS, input->b_mem});
    input->matmul_args.insert({DNNL_ARG_DST, input->c_mem});

#if ENALBE_BIAS
    input->matmul_args.insert({DNNL_ARG_BIAS, bias_mem});
#endif
#if ATTRIBUT_SCALE_ZP
    input->matmul_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale_mem});
    input->matmul_args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp_mem});
#endif

    return input;
}

void matmul_example(dnnl::engine::kind engine_kind) {
    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    memory::data_type type = memory::data_type::s8;
    bool is_integer
            = (type == memory::data_type::s8 || type == memory::data_type::u8);
    bool quick_test = false;
    int runs = 20;

    std::vector<std::shared_ptr<MM_Inputs>> vec_inputs;
    for (int i = 1; i < runs; i++) {
        int m = 1024, n = 1024, k = 768;
        int batch = i % 5 + 1;

        // 1x1x2048:1x2048x2048
        // batch = 1;
        memory::dims input_dims = {batch, 1, 2048};
        memory::dims weighs_dims = {1, 2048, 2048};
        memory::dims output_dims = {batch, 1, 2048};
        auto input = get_inputs(is_integer, type, engine, engine_stream, input_dims, weighs_dims, output_dims);

        vec_inputs.push_back(std::move(input));
    }

    // warmup.
    for (int i = 0; i< vec_inputs.size(); i++) {
        vec_inputs[i]->matmul_prim.execute(engine_stream, vec_inputs[i]->matmul_args);
        engine_stream.wait();
    }

    auto start_first = std::chrono::high_resolution_clock::now();
    for (int i = 0; i< vec_inputs.size(); i++) {
        vec_inputs[i]->matmul_prim.execute(engine_stream, vec_inputs[i]->matmul_args);
        engine_stream.wait();
    }
    auto end_first = std::chrono::high_resolution_clock::now();
    auto dur_first = std::chrono::duration_cast<std::chrono::microseconds>(end_first - start_first).count();

    // Timing runs.
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < vec_inputs.size(); i++)
    {
        std::cout << "Infer: [" << i << "], batch = " << vec_inputs[i]->a_mem.get_desc().get_dims()[0] << std::endl;
        vec_inputs[i]->matmul_prim.execute(engine_stream, vec_inputs[i]->matmul_args);
        engine_stream.wait();
    }

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // Display the result.
    double avg_time = duration / runs;
    // double total_ops = double(dims.m) * double(dims.n) * double(dims.k) * 2;
    // double perf = (total_ops / avg_time) * 1e-9;
    // auto scale_string = "G";
    // auto unit_string = is_integer ? "Op/s" : "Flop/s";
    // if (perf >= 1000) {
    //     perf /= 1000;
    //     scale_string = "T";
    // }
    // std::cout << perf << ' ' << scale_string << unit_string << std::endl;
    std::cout << "  avg_time = " << avg_time << "micro sec." << std::endl;
}

int getting_started_test_matmul(int argc, char **argv) {
    std::cout << "Start: matmul_example ..................." << std::endl;
    int exit_code = 0;

    matmul_example(dnnl::engine::kind::gpu);
    return exit_code;
}
