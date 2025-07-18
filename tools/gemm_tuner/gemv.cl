#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
#define GLOBAL_SIZE_DIM_2 \
    __private int global_size_dim0, __private int global_size_dim1,

#define GLOBAL_SIZE_DIM_3 \
    __private int global_size_dim0, __private int global_size_dim1, __private int global_size_dim2,

#define UNIFORM_BOUNDRY_CHECK_2(index0, index1) \
    if(index0 >= global_size_dim0 || index1 >= global_size_dim1) { \
        return; \
    }

#define UCHAR4_TO_CHAR8(b, scale, offset) \
    wei.s0 = (half)((b.s0 >> 4) - 8); \
    wei.s1 = (half)((b.s0 & 15) - 8); \
    wei.s2 = (half)((b.s1 >> 4) - 8); \
    wei.s3 = (half)((b.s1 & 15) - 8); \
    wei.s4 = (half)((b.s2 >> 4) - 8); \
    wei.s5 = (half)((b.s2 & 15) - 8); \
    wei.s6 = (half)((b.s3 >> 4) - 8); \
    wei.s7 = (half)((b.s3 & 15) - 8); \
    wei = wei * scale + offset;


#if WGS >= 8
__kernel void gemv_conv_c8_buf(GLOBAL_SIZE_DIM_3
                        __global const float* input,
                        __global const uchar *weight,
                        __global const float *dequantScaleOffset,
                        __global const float *bias,
                        __global float* output,
                        __private const int dstChannelAlign,
                        __private const int srcChannelAlign,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int srcChannel,
                        __private const int blockNum,
                        __private const int blockDim,
                        __private const float coef) {
    const int lid = get_local_id(0);
    const int oc = get_global_id(1); //oc/8
    const int oc8 = oc << 3;
    
    #if INPUT_CHANNEL_LEAVES_NUM != 0
    const int loop = max((srcChannel + 4 - 1) / 4 - 1, 0);
    #else
    const int loop = (srcChannel + 4 - 1) / 4;
    #endif
    const int weight_offset = oc * srcChannelC4 * 16;

    half8 out0 = 0;
    int input_offset = 0, output_offset = oc8;
    __local half8 sum0[WGS];
#ifdef COMPUTE_BATCH
    const int out_b_idx  = get_global_id(2) << 2; //b/4
    __local half8 sum1[WGS];
    __local half8 sum2[WGS];
    __local half8 sum3[WGS];
    half8 out1 = 0, out2 = 0, out3 = 0;
    input_offset = out_b_idx * srcChannelAlign;
    output_offset = oc8 + out_b_idx * dstChannelAlign;
#endif
    for(int j = lid; j < loop; j+=WGS){
        int k4 = j << 2;
        half8 scale = convert_half8(convert_float8(vload8(0, dequantScaleOffset + oc8 + (k4 / blockDim) * dstChannelC4 * 4)) / coef);
        half8 offset = 0;
        half8 wei;
        half4 in = convert_half4(vload4(0, input + k4 + input_offset));
        #ifdef COMPUTE_BATCH
        half4 in1 = convert_half4(vload4(0, input + input_offset + srcChannelAlign + k4));
        half4 in2 = convert_half4(vload4(0, input + input_offset + srcChannelAlign * 2 + k4));
        half4 in3 = convert_half4(vload4(0, input + input_offset + srcChannelAlign * 3 + k4));
        #endif
        uchar16 charWeightsInt40 = vload16(j, weight + weight_offset);
        {
            UCHAR4_TO_CHAR8(charWeightsInt40.s0123, scale, offset);
            out0 = mad((half8)in.s0, wei, out0);
            #ifdef COMPUTE_BATCH
            out1 = mad((half8)in1.s0, wei, out1);
            out2 = mad((half8)in2.s0, wei, out2);
            out3 = mad((half8)in3.s0, wei, out3);
            #endif
        }
        {
            UCHAR4_TO_CHAR8(charWeightsInt40.s4567, scale, offset);
            out0 = mad((half8)in.s1, wei, out0);
            #ifdef COMPUTE_BATCH
            out1 = mad((half8)in1.s1, wei, out1);
            out2 = mad((half8)in2.s1, wei, out2);
            out3 = mad((half8)in3.s1, wei, out3);
            #endif
        }
        {
            UCHAR4_TO_CHAR8(charWeightsInt40.s89ab, scale, offset);
            out0 = mad((half8)in.s2, wei, out0);
            #ifdef COMPUTE_BATCH
            out1 = mad((half8)in1.s2, wei, out1);
            out2 = mad((half8)in2.s2, wei, out2);
            out3 = mad((half8)in3.s2, wei, out3);
            #endif
        }
        {
            UCHAR4_TO_CHAR8(charWeightsInt40.scdef, scale, offset);
            out0 = mad((half8)in.s3, wei, out0);
            #ifdef COMPUTE_BATCH
            out1 = mad((half8)in1.s3, wei, out1);
            out2 = mad((half8)in2.s3, wei, out2);
            out3 = mad((half8)in3.s3, wei, out3);
            #endif
        }
    }
#if INPUT_CHANNEL_LEAVES_NUM != 0
    {
        int k4 = loop << 2;
        half8 scale = convert_half8(convert_float8(vload8(0, dequantScaleOffset + oc8 + (k4 / blockDim) * dstChannelC4 * 4)) / coef);
        half8 offset = 0;
        half8 wei;
        uchar16 charWeightsInt40 = vload16(loop, weight + weight_offset);
        {
            UCHAR4_TO_CHAR8(charWeightsInt40.s0123, scale, offset);
            out0 = mad((half8)input[k4], wei, out0);
        }
        #if INPUT_CHANNEL_LEAVES_NUM >= 2
        {
            UCHAR4_TO_CHAR8(charWeightsInt40.s4567, scale, offset);
            out0 = mad((half8)input[k4 + 1], wei, out0);
        }
        #endif
        #if INPUT_CHANNEL_LEAVES_NUM >= 3
        {
            UCHAR4_TO_CHAR8(charWeightsInt40.s89ab, scale, offset);
            out0 = mad((half8)input[k4 + 2], wei, out0);
        }
        #endif
    }
#endif
    sum0[lid] = out0;
    #ifdef COMPUTE_BATCH
    sum1[lid] = out1; sum2[lid] = out2; sum3[lid] = out3;
    #endif
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = WGS/2; i > 0; i /= 2){
        if (lid < i){
            sum0[lid] = sum0[lid] + sum0[lid + i];
            #ifdef COMPUTE_BATCH
            sum1[lid] = sum1[lid] + sum1[lid + i];
            sum2[lid] = sum2[lid] + sum2[lid + i];
            sum3[lid] = sum3[lid] + sum3[lid + i];
            #endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0){
        half8 vBias = convert_half8(vload8(0, bias + oc8));
        out0 = sum0[0] + vBias;

    #ifdef OUTPUT_CHANNEL_LEAVES
        vstore4(convert_float4(out0.s0123), 0, output + output_offset);
        if(oc8 + 4 < dstChannelC4 * 4)
            vstore4(convert_float4(out0.s4567), 0, output + 4 + output_offset);
    #else
        vstore8(convert_float8(out0), 0, output  + output_offset);
    #endif
    #ifdef COMPUTE_BATCH
        out1 = sum1[0] + vBias; out2 = sum2[0] + vBias; out3 = sum3[0] + vBias;
        #ifdef RELU
        out1 = fmax(out1, (half8)0);out2 = fmax(out2, (half8)0);out3 = fmax(out3, (half8)0);
        #endif
        #ifdef RELU6
        out1 = clamp(out1, (half8)0, (half8)6);out2 = clamp(out2, (half8)0, (half8)6);out3 = clamp(out3, (half8)0, (half8)6);
        #endif
        vstore8(convert_float8(out1), 0, output + output_offset + dstChannelAlign);
        vstore8(convert_float8(out2), 0, output + output_offset + dstChannelAlign + dstChannelAlign);
        vstore8(convert_float8(out3), 0, output + output_offset + dstChannelAlign + dstChannelAlign + dstChannelAlign);
    #endif
    }
}
#else
__kernel void gemv_conv_c8_buf(GLOBAL_SIZE_DIM_3
                        __global const float* input,
                        __global const uchar *weight,
                        __global const float *dequantScaleOffset,
                        __global const float *bias,
                        __global float* output,
                        __private const int dstChannelAlign,
                        __private const int srcChannelAlign,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int srcChannel,
                        __private const int blockNum,
                        __private const int blockDim,
                        __private const float coef) {
    const int ic = get_global_id(0);
    const int oc = get_global_id(1); //oc/8
    
    UNIFORM_BOUNDRY_CHECK_2(ic, oc);
    const int oc8 = oc << 3;

    const int loop = (blockDim + 4 - 1) / 4;
    #if INPUT_CHANNEL_LEAVES_NUM != 0
    const int loop_end = max(loop - 1, 0);
    #else
    const int loop_end = loop;
    #endif
    const int weight_offset = oc * srcChannelC4 * 16;
    half8 out0 = convert_half8(vload8(0, bias + oc8));
    for (int i = 0; i < blockNum; i++){
        half8 scale = convert_half8(convert_float8(vload8(0, dequantScaleOffset + oc8 + i * dstChannelC4 * 4)) / coef);
        half8 offset = 0;
        for (int j = 0; j < loop_end; j++) {
            int k = i * loop + j;
            half8 wei;
            half4 in = convert_half4(vload4(0, input + (k << 2)));
            #ifdef USE_IMAGE
            uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(k, oc)));
            #else
            uchar16 charWeightsInt40 = vload16(k, weight + weight_offset);
            #endif
            {
                UCHAR4_TO_CHAR8(charWeightsInt40.s0123, scale, offset);
                out0 = mad((half8)in.s0, wei, out0);
            }
            {
                UCHAR4_TO_CHAR8(charWeightsInt40.s4567, scale, offset);
                out0 = mad((half8)in.s1, wei, out0);
            }
            {
                UCHAR4_TO_CHAR8(charWeightsInt40.s89ab, scale, offset);
                out0 = mad((half8)in.s2, wei, out0);
            }
            {
                UCHAR4_TO_CHAR8(charWeightsInt40.scdef, scale, offset);
                out0 = mad((half8)in.s3, wei, out0);
            }
        }
        #if INPUT_CHANNEL_LEAVES_NUM != 0
        {
            int k = i * loop + loop_end;
            int k4 = k << 2;
            half8 wei;
            #ifdef USE_IMAGE
            uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(k, oc)));
            #else
            uchar16 charWeightsInt40 = vload16(k, weight + weight_offset);
            #endif
            {
                UCHAR4_TO_CHAR8(charWeightsInt40.s0123, scale, offset);
                out0 = mad((half8)input[k4], wei, out0);
            }
            #if INPUT_CHANNEL_LEAVES_NUM >= 2
            {
                UCHAR4_TO_CHAR8(charWeightsInt40.s4567, scale, offset);
                out0 = mad((half8)input[k4 + 1], wei, out0);
            }
            #endif
            #if INPUT_CHANNEL_LEAVES_NUM >= 3
            {
                UCHAR4_TO_CHAR8(charWeightsInt40.s89ab, scale, offset);
                out0 = mad((half8)input[k4 + 2], wei, out0);
            }
            #endif
        }
        #endif
    }
    #ifdef OUTPUT_CHANNEL_LEAVES
    vstore4(convert_float4(out0.s0123), 0, output + oc8);
    if(oc8 + 4 < dstChannelC4 * 4)
        vstore4(convert_float4(out0.s4567), 0, output + oc8 + 4);
    #else
    vstore8(convert_float8(out0), 0, output + oc8);
    #endif
}
#endif
