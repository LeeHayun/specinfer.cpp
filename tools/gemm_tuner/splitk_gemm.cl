#pragma OPENCL EXTENSION cl_khr_fp16 : enable
//#pragma OPENCL EXTENSION cl_khr_subgroups : enable

// #ifdef cl_qcom_reqd_sub_group_size
// #pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
// #define ADRENO_GPU 1
// #define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
// #endif

// assume
#define QK4_0 32
#define N_SIMDGROUP 4

#define WG_M_IDX get_group_id(DIM_M)
#define WG_N_IDX get_group_id(DIM_N)
#define WG_K_IDX 1
#define WI_M_IDX get_local_id(DIM_M)
#define WI_N_IDX get_local_id(DIM_N)
#define WI_K_IDX get_local_id(DIM_K)

#define M_IDX get_global_id(DIM_M)
#define N_IDX get_global_id(DIM_N)
#define K_IDX get_global_id(DIM_K)

#define WI_K_SIZE get_local_size(DIM_K)
#define LLID get_local_linear_id()

#define UCHAR4_TO_CHAR8(b, scale) \
    wei.s0 = (half)((b.s0 & 15) - 8); \
    wei.s1 = (half)((b.s0 >> 4) - 8); \
    wei.s2 = (half)((b.s1 & 15) - 8); \
    wei.s3 = (half)((b.s1 >> 4) - 8); \
    wei.s4 = (half)((b.s2 & 15) - 8); \
    wei.s5 = (half)((b.s2 >> 4) - 8); \
    wei.s6 = (half)((b.s3 & 15) - 8); \
    wei.s7 = (half)((b.s3 >> 4) - 8); \
    wei = wei * scale;


__kernel void kernel_gemv_conv_c8_buf(
        // __read_only  image1d_buffer_t src0_q,  // quantized A
        global uchar * src0_q,  // quantized A
        global half  * src0_d,  // A scales
        __read_only  image1d_buffer_t src1,    // B
        ulong offset1,            // offset to B (0)
        global float * dst,     // C
        ulong offsetd,            // offset to C (0)
        int ne00,               // K
        int ne01,               // M
        int ne02,               // 1
        int ne10,               // K
        int ne12,               // 1
        int ne0,                // M
        int ne1,                // N
        int r2,                 // 1
        int r3)
{

    const int oc = M_IDX;
    const int out_b_idx = N_IDX << 2;
    const int oc8 = oc << 3;

    const int N = ne1;
    const int K = ne00;
    const int M = ne01;
    const int K_4 = K / 4;

    const int input_offset = out_b_idx * K_4;
    const int output_offset = oc8 + out_b_idx * M;
    const int weight_offset = oc * K_4 * 16;

    // Assign sum value
    half8 out0 = 0, out1 = 0, out2 = 0, out3 = 0;
    #if WI_K > 1
    __local half8 sum0[WGS];
    __local half8 sum1[WGS];
    __local half8 sum2[WGS];
    __local half8 sum3[WGS];
    #endif

    for (int k = K_IDX; k < K_4; k += WI_K_SIZE) {
        half8 scale = vload8(0, src0_d + oc8 + (k / 8) * M);
        half8 wei;
        half4 in0 = read_imageh(src1, (input_offset + k));
        half4 in1 = 0, in2 = 0, in3 = 0;
        in1 = read_imageh(src1, (input_offset + k + K_4));
        in2 = read_imageh(src1, (input_offset + k + K_4 * 2));
        in3 = read_imageh(src1, (input_offset + k + K_4 * 3));
        
        uchar16 charWeights = vload16(k, src0_q + weight_offset);
        {
            UCHAR4_TO_CHAR8(charWeights.s0123, scale);
            out0 = mad((half8)in0.s0, wei, out0);
            out1 = mad((half8)in1.s0, wei, out1);
            out2 = mad((half8)in2.s0, wei, out2);
            out3 = mad((half8)in3.s0, wei, out3);
        }
        {
            UCHAR4_TO_CHAR8(charWeights.s4567, scale);
            out0 = mad((half8)in0.s1, wei, out0);
            out1 = mad((half8)in1.s1, wei, out1);
            out2 = mad((half8)in2.s1, wei, out2);
            out3 = mad((half8)in3.s1, wei, out3);
        }
        {
            UCHAR4_TO_CHAR8(charWeights.s89ab, scale);
            out0 = mad((half8)in0.s2, wei, out0);
            out1 = mad((half8)in1.s2, wei, out1);
            out2 = mad((half8)in2.s2, wei, out2);
            out3 = mad((half8)in3.s2, wei, out3);
        }
        {
            UCHAR4_TO_CHAR8(charWeights.scdef, scale);
            out0 = mad((half8)in0.s3, wei, out0);
            out1 = mad((half8)in1.s3, wei, out1);
            out2 = mad((half8)in2.s3, wei, out2);
            out3 = mad((half8)in3.s3, wei, out3);
        }
    }

    // reduce
    #if WI_K > 1
    sum0[LLID] = out0;
    sum1[LLID] = out1;
    sum2[LLID] = out2;
    sum3[LLID] = out3;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = WGS/2; i >= WGS/WI_K_SIZE; i /= 2) {
        if (LLID < i) {
            sum0[LLID] = sum0[LLID] + sum0[LLID + i];
            if (out_b_idx + 1 < N) sum1[LLID] = sum1[LLID] + sum1[LLID + i];
            if (out_b_idx + 2 < N) sum2[LLID] = sum2[LLID] + sum2[LLID + i];
            if (out_b_idx + 3 < N) sum3[LLID] = sum3[LLID] + sum3[LLID + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // output
    if (K_IDX == 0) {
        dst = (global float*)((global char*)dst + offsetd) + output_offset;
        vstore8(convert_float8(sum0[0]), 0, dst);
        if (out_b_idx + 1 < N) vstore8(convert_float8(sum1[0]), 0, dst + M);
        if (out_b_idx + 2 < N) vstore8(convert_float8(sum2[0]), 0, dst + M * 2);
        if (out_b_idx + 3 < N) vstore8(convert_float8(sum3[0]), 0, dst + M * 3);
    }
    #else
    dst = (global float*)((global char*)dst + offsetd) + output_offset;
    vstore8(convert_float8(out0), 0, dst);
    if (out_b_idx + 1 < N) vstore8(convert_float8(out1), 0, dst + M);
    if (out_b_idx + 2 < N) vstore8(convert_float8(out2), 0, dst + M * 2);
    if (out_b_idx + 3 < N) vstore8(convert_float8(out3), 0, dst + M * 3);
    #endif
}