#include "base.inc"

__kernel void LayerNormDim3Reduce1D(GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
                        __read_only image2d_t scale,
                        __read_only image2d_t eps, 
                        __private const int height,
                        __write_only image2d_t output) {
    const int width_idx    = get_global_id(0);
    const int chan_blk_idx = get_global_id(1);
    const int hb_idx       = get_global_id(2);

    if(!width_idx && !chan_blk_idx && !hb_idx) {
        int2 dim0 = get_image_dim(input);
        int2 dim1 = get_image_dim(scale);
        int2 dim2 = get_image_dim(eps);
        printf("input size: %d, %d\n", dim0.s0, dim0.s1);
        printf("scale size: %d, %d\n", dim1.s0, dim1.s1);
        printf("eps size: %d, %d\n", dim2.s0, dim2.s1);
    }
    DEAL_NON_UNIFORM_DIM3(width_idx, chan_blk_idx, hb_idx);
    const int width = global_size_dim0;

    int pos = mad24(chan_blk_idx, width, width_idx);

    float4 data   = read_imagef(input, SAMPLER, (int2)(pos, hb_idx));
    float4 scale_ = read_imagef(scale, SAMPLER, (int2)(0, hb_idx % height));
    float4 eps_   = read_imagef(eps, SAMPLER, (int2)(0, hb_idx % height)); 
    // float4 scale_ = read_imagef(scale, SAMPLER, (int2)(0, hb_idx));
    // float4 eps_   = read_imagef(eps, SAMPLER, (int2)(0, hb_idx)); 
    data          = mad(data, scale_, eps_);

    write_imagef(output, (int2)(pos, hb_idx), data);
}
