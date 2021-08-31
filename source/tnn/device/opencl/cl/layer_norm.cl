#include "base.inc"

// __kernel void LayerNormDim3Reduce1D(GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
//                         __read_only image2d_t scale,
//                         __read_only image2d_t eps, 
//                         __private const int height,
//                         __write_only image2d_t output) {
//     const int width_idx    = get_global_id(0);
//     const int chan_blk_idx = get_global_id(1);
//     const int hb_idx       = get_global_id(2);

    
//     DEAL_NON_UNIFORM_DIM3(width_idx, chan_blk_idx, hb_idx);
//     const int width = global_size_dim0;

//     int pos = mad24(chan_blk_idx, width, width_idx);

//     float4 data   = read_imagef(input, SAMPLER, (int2)(pos, hb_idx));
//     float4 scale_ = (FLOAT4)read_imagef(scale, SAMPLER, (int2)(0, hb_idx % height)).s0;
//     float4 eps_   = (FLOAT4)read_imagef(eps, SAMPLER, (int2)(0, hb_idx % height)).s0;
//     if(!width_idx && !chan_blk_idx && !hb_idx) {
//         int2 dim0 = get_image_dim(input);
//         int2 dim1 = get_image_dim(scale);
//         int2 dim2 = get_image_dim(eps);
//         float4 data1   = read_imagef(input, SAMPLER, (int2)(0, 128));
//         printf("input size: %d, %d\n", dim0.s0, dim0.s1);
//         printf("scale size: %d, %d\n", dim1.s0, dim1.s1);
//         printf("eps size: %d, %d\n", dim2.s0, dim2.s1);
//         printf("eps_: %f, %f, %f, %f\n", eps_.s0, eps_.s1, eps_.s2, eps_.s3);
//         printf("data: %f, %f, %f, %f\n", data.s0, data.s1, data.s2, data.s3);
//         printf("data1: %f, %f, %f, %f\n", data1.s0, data1.s1, data1.s2, data1.s3);
//     }
//     // if(!width_idx && !chan_blk_idx && hb_idx == 1) {
//     //     printf("data1: %f, %f, %f, %f\n", data.s0, data.s1, data.s2, data.s3);
//     // }
//     // float4 scale_ = read_imagef(scale, SAMPLER, (int2)(0, hb_idx));
//     // float4 eps_   = read_imagef(eps, SAMPLER, (int2)(0, hb_idx)); 
//     data          = mad(data, scale_, eps_);

//     write_imagef(output, (int2)(pos, hb_idx), data);
// }


__kernel void LayerNormDim3Reduce1D(GLOBAL_SIZE_3_DIMS __read_only image2d_t input, __read_only image2d_t scale,
                                    __read_only image2d_t bias, __private const float eps, __private const int height,
                                    __write_only image2d_t output) {
    const int width_idx    = get_global_id(0);
    const int chan_blk_idx = get_global_id(1);
    const int hb_idx       = get_global_id(2);

    // if (!width_idx && !chan_blk_idx && !hb_idx) {
    //     int2 dim0 = get_image_dim(input);
    //     int2 dim1 = get_image_dim(scale);
    //     int2 dim2 = get_image_dim(bias);
    // }
    DEAL_NON_UNIFORM_DIM3(width_idx, chan_blk_idx, hb_idx);
    const int width     = global_size_dim0;
    const int out_h_idx = hb_idx / height;

    int pos = mad24(chan_blk_idx, width, width_idx);

    FLOAT4 in;
    FLOAT4 mean  = (FLOAT4)0;
    FLOAT4 var   = (FLOAT4)0;
    int hb_start = out_h_idx * height;
    for (int h = 0; h < height; h++) {
        in = RI_F(input, SAMPLER, (int2)(pos, hb_start + h));
        mean += in;
        var += (in * in);
    }
    mean /= height;
    var = var / height - mean * mean;
    var = 1.0f / sqrt(var + eps);

    FLOAT4 data   = RI_F(input, SAMPLER, (int2)(pos, hb_idx));
    FLOAT4 scale_ = (FLOAT4)RI_F(scale, SAMPLER, (int2)(0, hb_idx % height)).s0;
    FLOAT4 bias_   = (FLOAT4)RI_F(bias, SAMPLER, (int2)(0, hb_idx % height)).s0;
    // FLOAT4 scale_ = RI_F(scale, SAMPLER, (int2)(chan_blk_idx, 0));
    // FLOAT4 bias_  = RI_F(bias, SAMPLER, (int2)(chan_blk_idx, 0));

    data -= mean;
    scale_ *= var;
    data = mad(data, scale_, bias_);

    write_imagef(output, (int2)(pos, hb_idx), data);
}
