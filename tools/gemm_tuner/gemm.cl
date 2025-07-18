#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define GLOBAL_SIZE_DIM2 \
    __private int global_size_dim0, __private int global_size_dim1,

#define UNIFORM_BOUNDRY_CHECK(index0, index1) \
    if(index0 >= global_size_dim0 || index1 >= global_size_dim1) { \
        return; \
    }

#define UCHAR4_TO_CHAR8(a, c) \
    a.s0 = (c.s0 >> 4) - 8; a.s1 = (c.s0 & 15) - 8; a.s2 = (c.s1 >> 4) - 8; a.s3 = (c.s1 & 15) - 8; a.s4 = (c.s2 >> 4) - 8; a.s5 = (c.s2 & 15) - 8; a.s6 = (c.s3 >> 4) - 8; a.s7 = (c.s3 & 15) - 8;

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;


#define UCHAR4_TO_FLOAT8(b, scale, offset) \
    wei.s0 = (half)((b.s0 >> 4) - 8); \
    wei.s1 = (half)((b.s0 & 15) - 8); \
    wei.s2 = (half)((b.s1 >> 4) - 8); \
    wei.s3 = (half)((b.s1 & 15) - 8); \
    wei.s4 = (half)((b.s2 >> 4) - 8); \
    wei.s5 = (half)((b.s2 & 15) - 8); \
    wei.s6 = (half)((b.s3 >> 4) - 8); \
    wei.s7 = (half)((b.s3 & 15) - 8); \
    wei = wei * scale + offset;

__kernel void gemm_b4_c8_int4_buf(GLOBAL_SIZE_DIM2
                        __global const float* input,
                        __global const uchar *weight,
                        __global const float *dequantScaleOffset,
                        __global const float *bias,
                        __global float* output,
                        __private const int bhw,
                        __private const int dstChannelAlign,
                        __private const int srcChannelAlign,
                        __private const int blockNum,
                        __private const int blockDim,
                        __private const float coef) {
    const int x = get_global_id(0); //b/4
    const int y  = get_global_id(1); //c/8

    UNIFORM_BOUNDRY_CHECK(x, y);
    
    const int out_b_idx = x << 2;
    const int out_c_idx = y << 1;

    half8 out0 = convert_half8(vload8(0, bias + (out_c_idx << 2)));
    half8 out1 = out0;
    half8 out2 = out0;
    half8 out3 = out0;
    
    const int bhw4 = bhw << 2;
    const int input_offset = out_b_idx * 4;
    int out_offset = out_c_idx * bhw4 + out_b_idx * 4;
    const int weight_offset = y * srcChannelAlign * 4;
    const int loop = (blockDim + 4 - 1) / 4;
#if INPUT_CHANNEL_LEAVES_NUM != 0
    const int loop_end = max(loop - 1, 0);
#else
    const int loop_end = loop;
#endif

#if INPUT_BATCH_LEAVES_NUM != 0
    if(out_b_idx + 3 >= bhw){
        for (int i = 0; i < blockNum; i++){
            half8 scale = convert_float8(convert_half8(vload8(0, dequantScaleOffset + (out_c_idx << 2) + i * dstChannelAlign)) / coef);
            half8 offset = 0;
            for (int j = 0; j < loop_end; j++) {
                int k = i * loop + j;
                half8 wei;
                #ifdef USE_IMAGE
                uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(k, y)));
                #else
                uchar16 charWeightsInt40 = vload16(k, weight + weight_offset);
                #endif
                half4 in0 = convert_float4(vload4(0, input + input_offset + k * bhw4));
                #if INPUT_BATCH_LEAVES_NUM >= 2
                half4 in1 = convert_float4(vload4(0, input + input_offset + k * bhw4 + 4));
                #endif
                #if INPUT_BATCH_LEAVES_NUM >= 3
                half4 in2 = convert_float4(vload4(0, input + input_offset + k * bhw4 + 8));
                #endif
                {
                    UCHAR4_TO_FLOAT8(charWeightsInt40.s0123, scale, offset);
                    out0 = mad((half8)in0.s0, wei, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((half8)in1.s0, wei, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((half8)in2.s0, wei, out2);
                    #endif
                }
                {
                    UCHAR4_TO_FLOAT8(charWeightsInt40.s4567, scale, offset);
                    out0 = mad((half8)in0.s1, wei, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((half8)in1.s1, wei, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((half8)in2.s1, wei, out2);
                    #endif
                }
                {
                    UCHAR4_TO_FLOAT8(charWeightsInt40.s89ab, scale, offset);
                    out0 = mad((half8)in0.s2, wei, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((half8)in1.s2, wei, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((half8)in2.s2, wei, out2);
                    #endif
                }
                {
                    UCHAR4_TO_FLOAT8(charWeightsInt40.scdef, scale, offset);
                    out0 = mad((half8)in0.s3, wei, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((half8)in1.s3, wei, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((half8)in2.s3, wei, out2);
                    #endif
                }
            }
            #if INPUT_CHANNEL_LEAVES_NUM != 0
            {
                int k = i * loop + loop_end;
                half8 wei;
                half4 in0 = convert_float4(vload4(0, input + input_offset + k * bhw4));
                #if INPUT_BATCH_LEAVES_NUM >= 2
                half4 in1 = convert_float4(vload4(0, input + input_offset + k * bhw4 + 4));
                #endif
                #if INPUT_BATCH_LEAVES_NUM >= 3
                half4 in2 = convert_float4(vload4(0, input + input_offset + k * bhw4 + 8));
                #endif
                #ifdef USE_IMAGE
                uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(k, y)));
                #else
                uchar16 charWeightsInt40 = vload16(k, weight + weight_offset);
                #endif
                {
                    UCHAR4_TO_FLOAT8(charWeightsInt40.s0123, scale, offset);
                    out0 = mad((half8)in0.s0, wei, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((half8)in1.s0, wei, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((half8)in2.s0, wei, out2);
                    #endif
                }
                #if INPUT_CHANNEL_LEAVES_NUM >= 2
                {
                    UCHAR4_TO_FLOAT8(charWeightsInt40.s4567, scale, offset);
                    out0 = mad((half8)in0.s1, wei, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((half8)in1.s1, wei, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((half8)in2.s1, wei, out2);
                    #endif
                }
                #endif
                #if INPUT_CHANNEL_LEAVES_NUM >= 3
                {
                    UCHAR4_TO_FLOAT8(charWeightsInt40.s89ab, scale, offset);
                    out0 = mad((half8)in0.s2, wei, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((half8)in1.s2, wei, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((half8)in2.s2, wei, out2);
                    #endif
                }
                #endif
            }
            #endif
        }
    } else {
#endif
    for (int i = 0; i < blockNum; i++){
        half8 scale = convert_half8(convert_float8(vload8(0, dequantScaleOffset + (out_c_idx << 2) + i * dstChannelAlign)) / coef);
        half8 offset = 0;
        for (int j = 0; j < loop_end; j++) {
            int k = i * loop + j;
            half8 wei;
            half16 in = convert_half16(vload16(0, input + input_offset + k * bhw4));
            #ifdef USE_IMAGE
            uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(k, y)));
            #else
            uchar16 charWeightsInt40 = vload16(k, weight + weight_offset);
            #endif
            {
                UCHAR4_TO_FLOAT8(charWeightsInt40.s0123, scale, offset);
                out0 = mad((half8)in.s0, wei, out0);
                out1 = mad((half8)in.s4, wei, out1);
                out2 = mad((half8)in.s8, wei, out2);
                out3 = mad((half8)in.sc, wei, out3);
            }
            {
                UCHAR4_TO_FLOAT8(charWeightsInt40.s4567, scale, offset);
                out0 = mad((half8)in.s1, wei, out0);
                out1 = mad((half8)in.s5, wei, out1);
                out2 = mad((half8)in.s9, wei, out2);
                out3 = mad((half8)in.sd, wei, out3);
            }
            {
                UCHAR4_TO_FLOAT8(charWeightsInt40.s89ab, scale, offset);
                out0 = mad((half8)in.s2, wei, out0);
                out1 = mad((half8)in.s6, wei, out1);
                out2 = mad((half8)in.sa, wei, out2);
                out3 = mad((half8)in.se, wei, out3);
            }
            {
                UCHAR4_TO_FLOAT8(charWeightsInt40.scdef, scale, offset);
                out0 = mad((half8)in.s3, wei, out0);
                out1 = mad((half8)in.s7, wei, out1);
                out2 = mad((half8)in.sb, wei, out2);
                out3 = mad((half8)in.sf, wei, out3);
            }
        }
        #if INPUT_CHANNEL_LEAVES_NUM != 0
        {
            int k = i * loop + loop_end;
            half8 wei;
            half16 in = convert_half16(vload16(0, input + input_offset + k * bhw4));
            #ifdef USE_IMAGE
            uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(k, y)));
            #else
            uchar16 charWeightsInt40 = vload16(k, weight + weight_offset);
            #endif
            {
                UCHAR4_TO_FLOAT8(charWeightsInt40.s0123, scale, offset);
                out0 = mad((half8)in.s0, wei, out0);
                out1 = mad((half8)in.s4, wei, out1);
                out2 = mad((half8)in.s8, wei, out2);
                out3 = mad((half8)in.sc, wei, out3);
            }
            #if INPUT_CHANNEL_LEAVES_NUM >= 2
            {
                UCHAR4_TO_FLOAT8(charWeightsInt40.s4567, scale, offset);
                out0 = mad((half8)in.s1, wei, out0);
                out1 = mad((half8)in.s5, wei, out1);
                out2 = mad((half8)in.s9, wei, out2);
                out3 = mad((half8)in.sd, wei, out3);
            }
            #endif
            #if INPUT_CHANNEL_LEAVES_NUM >= 3
            {
                UCHAR4_TO_FLOAT8(charWeightsInt40.s89ab, scale, offset);
                out0 = mad((half8)in.s2, wei, out0);
                out1 = mad((half8)in.s6, wei, out1);
                out2 = mad((half8)in.sa, wei, out2);
                out3 = mad((half8)in.se, wei, out3);
            }
            #endif
        }
        #endif
    }
#if INPUT_BATCH_LEAVES_NUM != 0
    }
#endif

#if INPUT_BATCH_LEAVES_NUM != 0
    if(out_b_idx + 3 >= bhw){
        #if INPUT_BATCH_LEAVES_NUM == 3
        vstore8(convert_float8((half8)(out0.s0123, out1.s0123)), 0, output+out_offset);
        vstore4(convert_float4(out2.s0123), 0, output+out_offset+8);
        if((out_c_idx << 2) + 4 < dstChannelAlign){
            vstore8(convert_float8((half8)(out0.s4567, out1.s4567)), 0, output+out_offset+bhw4);
            vstore4(convert_float4(out2.s4567), 0, output+out_offset+bhw4+8);
        }
        #elif INPUT_BATCH_LEAVES_NUM == 2
        vstore8(convert_float8((half8)(out0.s0123, out1.s0123)), 0, output+out_offset);
        if((out_c_idx << 2) + 4 < dstChannelAlign){
            vstore8(convert_float8((half8)(out0.s4567, out1.s4567)), 0, output+out_offset+bhw4);
        }
        #elif INPUT_BATCH_LEAVES_NUM == 1
        vstore4(convert_float4(out0.s0123), 0, output+out_offset);
        if((out_c_idx << 2) + 4 < dstChannelAlign){
            vstore4(convert_float4(out0.s4567), 0, output+out_offset+bhw4);
        }
        #endif
    }else{
#endif
        vstore16(convert_float16((half16)(out0.s0123, out1.s0123, out2.s0123, out3.s0123)), 0, output+out_offset);
        if((out_c_idx << 2) + 4 < dstChannelAlign){
            vstore16(convert_float16((half16)(out0.s4567, out1.s4567, out2.s4567, out3.s4567)), 0, output+out_offset+bhw4);
        }
#if INPUT_BATCH_LEAVES_NUM != 0
    }
#endif
}
