// Auto-generated from general_gemm.cl
// DO NOT EDIT - 이 파일은 자동 생성되었습니다.

#ifndef GENERAL_GEMM_H
#define GENERAL_GEMM_H

#include <string>

namespace OpenCLKernels {
    const std::string GENERAL_GEMM_KERNEL_SOURCE = R"(
// src0_q, src0_d, src1 are transposed as a preprocessing step
// 4-bit weights are transposed in groups of 4 (unsigned short int)
// consider weights originally "next to each other", now "on top of each other"
// each fiber computes a 8x4 tile of output elements
// using unshuffled weights

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// #pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

// #ifdef cl_qcom_reqd_sub_group_size
// #pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
// #define ADRENO_GPU 1
// #define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
// #endif

// #ifdef ADRENO_GPU
// REQD_SUBGROUP_SIZE_128
// #endif

// #define DIM_M 0
// #define DIM_N 1
// #define DIM_K 2
// #define TILE_M 8
// #define TILE_N 4
// #define TILE_K 1

// Need to define DIM_M, DIM_N, DIM_K, TILE_M, TILE_N, TILE_K
#define WG_M_IDX get_group_id(DIM_M)
#define WG_N_IDX get_group_id(DIM_N)
#define WG_K_IDX 1
#define WI_M_IDX get_local_id(DIM_M)
#define WI_N_IDX get_local_id(DIM_N)
#define WI_K_IDX get_local_id(DIM_K)

#define WG_M_SIZE get_num_groups(DIM_M)
#define WG_N_SIZE get_num_groups(DIM_N)
#define WG_K_SIZE 1
#define WI_M_SIZE get_local_size(DIM_M)
#define WI_N_SIZE get_local_size(DIM_N)
#define WI_K_SIZE get_local_size(DIM_K)

// #define GLOBAL_M_SIZE get_global_size(DIM_M)
// #define GLOBAL_N_SIZE get_global_size(DIM_N)
// #define GLOBAL_K_SIZE get_global_size(DIM_K)
#define GLOBAL_M_IDX get_global_id(DIM_M)
#define GLOBAL_N_IDX get_global_id(DIM_N)
#define GLOBAL_K_IDX get_global_id(DIM_K)

// #define WEIGHT_TILE_SIZE (TILE_M * TILE_K)
// #define WEIGHT_TILE_OFFSET (GLOBAL_M_SIZE * GLOBAL_K_SIZE * WEIGHT_TILE_SIZE)


#if TILE_M == 2
    #define DEFINE_C(n_idx) \
        half2 C##n_idx = 0;

    #if TILE_K == 8
        #define DEQUANTIZE_A_K0 \
            A.s0 = ((A_q.s0 & (0x000F)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s2 & (0x000F)) - 8) * A_d.s1;
            
        #define DEQUANTIZE_A_K1 \
            A.s0 = ((A_q.s0 & (0x00F0)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s2 & (0x00F0)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K2 \
            A.s0 = ((A_q.s0 & (0x0F00)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s2 & (0x0F00)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K3 \
            A.s0 = ((A_q.s0 & (0xF000)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s2 & (0xF000)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K4 \
            A.s0 = ((A_q.s1 & (0x000F)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s3 & (0x000F)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K5 \
            A.s0 = ((A_q.s1 & (0x00F0)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s3 & (0x00F0)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K6 \
            A.s0 = ((A_q.s1 & (0x0F00)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s3 & (0x0F00)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K7 \
            A.s0 = ((A_q.s1 & (0xF000)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s3 & (0xF000)) - 8) * A_d.s1;

    #elif TILE_K == 16
        #define DEQUANTIZE_A_K0 \
            A.s0 = ((A_q.s0 & (0x000F)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s4 & (0x000F)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K1 \
            A.s0 = ((A_q.s0 & (0x00F0)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s4 & (0x00F0)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K2 \
            A.s0 = ((A_q.s0 & (0x0F00)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s4 & (0x0F00)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K3 \
            A.s0 = ((A_q.s0 & (0xF000)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s4 & (0xF000)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K4 \
            A.s0 = ((A_q.s1 & (0x000F)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s5 & (0x000F)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K5 \
            A.s0 = ((A_q.s1 & (0x00F0)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s5 & (0x00F0)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K6 \
            A.s0 = ((A_q.s1 & (0x0F00)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s5 & (0x0F00)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K7 \
            A.s0 = ((A_q.s1 & (0xF000)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s5 & (0xF000)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K8 \
            A.s0 = ((A_q.s2 & (0x000F)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s6 & (0x000F)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K9 \
            A.s0 = ((A_q.s2 & (0x00F0)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s6 & (0x00F0)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K10 \
            A.s0 = ((A_q.s2 & (0x0F00)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s6 & (0x0F00)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K11 \
            A.s0 = ((A_q.s2 & (0xF000)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s6 & (0xF000)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K12 \
            A.s0 = ((A_q.s3 & (0x000F)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s7 & (0x000F)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K13 \
            A.s0 = ((A_q.s3 & (0x00F0)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s7 & (0x00F0)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K14 \
            A.s0 = ((A_q.s3 & (0x0F00)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s7 & (0x0F00)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K15 \
            A.s0 = ((A_q.s3 & (0xF000)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s7 & (0xF000)) - 8) * A_d.s1;

    #elif TILE_K == 32
        #define DEQUANTIZE_A_K0 \
            A.s0 = ((A_q.s0 & (0x000F)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s8 & (0x000F)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K1 \
            A.s0 = ((A_q.s0 & (0x00F0)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s8 & (0x00F0)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K2 \
            A.s0 = ((A_q.s0 & (0x0F00)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s8 & (0x0F00)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K3 \
            A.s0 = ((A_q.s0 & (0xF000)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s8 & (0xF000)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K4 \
            A.s0 = ((A_q.s1 & (0x000F)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s9 & (0x000F)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K5 \
            A.s0 = ((A_q.s1 & (0x00F0)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s9 & (0x00F0)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K6 \
            A.s0 = ((A_q.s1 & (0x0F00)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s9 & (0x0F00)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K7 \
            A.s0 = ((A_q.s1 & (0xF000)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s9 & (0xF000)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K8 \
            A.s0 = ((A_q.s2 & (0x000F)) - 8) * A_d.s0; \
            A.s1 = ((A_q.sA & (0x000F)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K9 \
            A.s0 = ((A_q.s2 & (0x00F0)) - 8) * A_d.s0; \
            A.s1 = ((A_q.sA & (0x00F0)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K10 \
            A.s0 = ((A_q.s2 & (0x0F00)) - 8) * A_d.s0; \
            A.s1 = ((A_q.sA & (0x0F00)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K11 \
            A.s0 = ((A_q.s2 & (0xF000)) - 8) * A_d.s0; \
            A.s1 = ((A_q.sA & (0xF000)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K12 \
            A.s0 = ((A_q.s3 & (0x000F)) - 8) * A_d.s0; \
            A.s1 = ((A_q.sB & (0x000F)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K13 \
            A.s0 = ((A_q.s3 & (0x00F0)) - 8) * A_d.s0; \
            A.s1 = ((A_q.sB & (0x00F0)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K14 \
            A.s0 = ((A_q.s3 & (0x0F00)) - 8) * A_d.s0; \
            A.s1 = ((A_q.sB & (0x0F00)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K15 \
            A.s0 = ((A_q.s3 & (0xF000)) - 8) * A_d.s0; \
            A.s1 = ((A_q.sB & (0xF000)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K16 \
            A.s0 = ((A_q.s4 & (0x000F)) - 8) * A_d.s0; \
            A.s1 = ((A_q.sC & (0x000F)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K17 \
            A.s0 = ((A_q.s4 & (0x00F0)) - 8) * A_d.s0; \
            A.s1 = ((A_q.sC & (0x00F0)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K18 \
            A.s0 = ((A_q.s4 & (0x0F00)) - 8) * A_d.s0; \
            A.s1 = ((A_q.sC & (0x0F00)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K19 \
            A.s0 = ((A_q.s4 & (0xF000)) - 8) * A_d.s0; \
            A.s1 = ((A_q.sC & (0xF000)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K20 \
            A.s0 = ((A_q.s5 & (0x000F)) - 8) * A_d.s0; \
            A.s1 = ((A_q.sD & (0x000F)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K21 \
            A.s0 = ((A_q.s5 & (0x00F0)) - 8) * A_d.s0; \
            A.s1 = ((A_q.sD & (0x00F0)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K22 \
            A.s0 = ((A_q.s5 & (0x0F00)) - 8) * A_d.s0; \
            A.s1 = ((A_q.sD & (0x0F00)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K23 \
            A.s0 = ((A_q.s5 & (0xF000)) - 8) * A_d.s0; \
            A.s1 = ((A_q.sD & (0xF000)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K24 \
            A.s0 = ((A_q.s6 & (0x000F)) - 8) * A_d.s0; \
            A.s1 = ((A_q.sE & (0x000F)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K25 \
            A.s0 = ((A_q.s6 & (0x00F0)) - 8) * A_d.s0; \
            A.s1 = ((A_q.sE & (0x00F0)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K26 \
            A.s0 = ((A_q.s6 & (0x0F00)) - 8) * A_d.s0; \
            A.s1 = ((A_q.sE & (0x0F00)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K27 \
            A.s0 = ((A_q.s6 & (0xF000)) - 8) * A_d.s0; \
            A.s1 = ((A_q.sE & (0xF000)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K28 \
            A.s0 = ((A_q.s7 & (0x000F)) - 8) * A_d.s0; \
            A.s1 = ((A_q.sF & (0x000F)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K29 \
            A.s0 = ((A_q.s7 & (0x00F0)) - 8) * A_d.s0; \
            A.s1 = ((A_q.sF & (0x00F0)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K30 \
            A.s0 = ((A_q.s7 & (0x0F00)) - 8) * A_d.s0; \
            A.s1 = ((A_q.sF & (0x0F00)) - 8) * A_d.s1;

        #define DEQUANTIZE_A_K31 \
            A.s0 = ((A_q.s7 & (0xF000)) - 8) * A_d.s0; \
            A.s1 = ((A_q.sF & (0xF000)) - 8) * A_d.s1;

    #endif

#elif TILE_M == 4
    #define DEFINE_C(n_idx) \
        half4 C##n_idx = 0;

    #if TILE_K == 4
        #define DEQUANTIZE_A_K0 \
            A.s0 = ((A_q.s0 & (0x000F)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s1 & (0x000F)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s2 & (0x000F)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s3 & (0x000F)) - 8) * A_d.s3;

        #define DEQUANTIZE_A_K1 \
            A.s0 = ((A_q.s0 & (0x00F0)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s1 & (0x00F0)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s2 & (0x00F0)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s3 & (0x00F0)) - 8) * A_d.s3;

        #define DEQUANTIZE_A_K2 \
            A.s0 = ((A_q.s0 & (0x0F00)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s1 & (0x0F00)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s2 & (0x0F00)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s3 & (0x0F00)) - 8) * A_d.s3;

        #define DEQUANTIZE_A_K3 \
            A.s0 = ((A_q.s0 & (0xF000)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s1 & (0xF000)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s2 & (0xF000)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s3 & (0xF000)) - 8) * A_d.s3;

    #elif TILE_K == 8
        #define DEQUANTIZE_A_K0 \
            A.s0 = ((A_q.s0 & (0x000F)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s2 & (0x000F)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s4 & (0x000F)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s6 & (0x000F)) - 8) * A_d.s3;

        #define DEQUANTIZE_A_K1 \
            A.s0 = ((A_q.s0 & (0x00F0)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s2 & (0x00F0)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s4 & (0x00F0)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s6 & (0x00F0)) - 8) * A_d.s3;

        #define DEQUANTIZE_A_K2 \
            A.s0 = ((A_q.s0 & (0x0F00)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s2 & (0x0F00)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s4 & (0x0F00)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s6 & (0x0F00)) - 8) * A_d.s3;

        #define DEQUANTIZE_A_K3 \
            A.s0 = ((A_q.s0 & (0xF000)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s2 & (0xF000)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s4 & (0xF000)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s6 & (0xF000)) - 8) * A_d.s3;

        #define DEQUANTIZE_A_K4 \
            A.s0 = ((A_q.s1 & (0x000F)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s3 & (0x000F)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s5 & (0x000F)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s7 & (0x000F)) - 8) * A_d.s3;

        #define DEQUANTIZE_A_K5 \
            A.s0 = ((A_q.s1 & (0x00F0)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s3 & (0x00F0)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s5 & (0x00F0)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s7 & (0x00F0)) - 8) * A_d.s3;

        #define DEQUANTIZE_A_K6 \
            A.s0 = ((A_q.s1 & (0x0F00)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s3 & (0x0F00)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s5 & (0x0F00)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s7 & (0x0F00)) - 8) * A_d.s3;

        #define DEQUANTIZE_A_K7 \
            A.s0 = ((A_q.s1 & (0xF000)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s3 & (0xF000)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s5 & (0xF000)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s7 & (0xF000)) - 8) * A_d.s3;

    #elif TILE_K == 16
        #define DEQUANTIZE_A_K0 \
            A.s0 = ((A_q.s0 & (0x000F)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s4 & (0x000F)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s8 & (0x000F)) - 8) * A_d.s2; \
            A.s3 = ((A_q.sC & (0x000F)) - 8) * A_d.s3;

        #define DEQUANTIZE_A_K1 \
            A.s0 = ((A_q.s0 & (0x00F0)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s4 & (0x00F0)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s8 & (0x00F0)) - 8) * A_d.s2; \
            A.s3 = ((A_q.sC & (0x00F0)) - 8) * A_d.s3;

        #define DEQUANTIZE_A_K2 \
            A.s0 = ((A_q.s0 & (0x0F00)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s4 & (0x0F00)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s8 & (0x0F00)) - 8) * A_d.s2; \
            A.s3 = ((A_q.sC & (0x0F00)) - 8) * A_d.s3;

        #define DEQUANTIZE_A_K3 \
            A.s0 = ((A_q.s0 & (0xF000)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s4 & (0xF000)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s8 & (0xF000)) - 8) * A_d.s2; \
            A.s3 = ((A_q.sC & (0xF000)) - 8) * A_d.s3;

        #define DEQUANTIZE_A_K4 \
            A.s0 = ((A_q.s1 & (0x000F)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s5 & (0x000F)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s9 & (0x000F)) - 8) * A_d.s2; \
            A.s3 = ((A_q.sD & (0x000F)) - 8) * A_d.s3;

        #define DEQUANTIZE_A_K5 \
            A.s0 = ((A_q.s1 & (0x00F0)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s5 & (0x00F0)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s9 & (0x00F0)) - 8) * A_d.s2; \
            A.s3 = ((A_q.sD & (0x00F0)) - 8) * A_d.s3;

        #define DEQUANTIZE_A_K6 \
            A.s0 = ((A_q.s1 & (0x0F00)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s5 & (0x0F00)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s9 & (0x0F00)) - 8) * A_d.s2; \
            A.s3 = ((A_q.sD & (0x0F00)) - 8) * A_d.s3;

        #define DEQUANTIZE_A_K7 \
            A.s0 = ((A_q.s1 & (0xF000)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s5 & (0xF000)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s9 & (0xF000)) - 8) * A_d.s2; \
            A.s3 = ((A_q.sD & (0xF000)) - 8) * A_d.s3;

        #define DEQUANTIZE_A_K8 \
            A.s0 = ((A_q.s2 & (0x000F)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s6 & (0x000F)) - 8) * A_d.s1; \
            A.s2 = ((A_q.sA & (0x000F)) - 8) * A_d.s2; \
            A.s3 = ((A_q.sE & (0x000F)) - 8) * A_d.s3;

        #define DEQUANTIZE_A_K9 \
            A.s0 = ((A_q.s2 & (0x00F0)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s6 & (0x00F0)) - 8) * A_d.s1; \
            A.s2 = ((A_q.sA & (0x00F0)) - 8) * A_d.s2; \
            A.s3 = ((A_q.sE & (0x00F0)) - 8) * A_d.s3;

        #define DEQUANTIZE_A_K10 \
            A.s0 = ((A_q.s2 & (0x0F00)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s6 & (0x0F00)) - 8) * A_d.s1; \
            A.s2 = ((A_q.sA & (0x0F00)) - 8) * A_d.s2; \
            A.s3 = ((A_q.sE & (0x0F00)) - 8) * A_d.s3;

        #define DEQUANTIZE_A_K11 \
            A.s0 = ((A_q.s2 & (0xF000)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s6 & (0xF000)) - 8) * A_d.s1; \
            A.s2 = ((A_q.sA & (0xF000)) - 8) * A_d.s2; \
            A.s3 = ((A_q.sE & (0xF000)) - 8) * A_d.s3;

        #define DEQUANTIZE_A_K12 \
            A.s0 = ((A_q.s3 & (0x000F)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s7 & (0x000F)) - 8) * A_d.s1; \
            A.s2 = ((A_q.sB & (0x000F)) - 8) * A_d.s2; \
            A.s3 = ((A_q.sF & (0x000F)) - 8) * A_d.s3;

        #define DEQUANTIZE_A_K13 \
            A.s0 = ((A_q.s3 & (0x00F0)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s7 & (0x00F0)) - 8) * A_d.s1; \
            A.s2 = ((A_q.sB & (0x00F0)) - 8) * A_d.s2; \
            A.s3 = ((A_q.sF & (0x00F0)) - 8) * A_d.s3;

        #define DEQUANTIZE_A_K14 \
            A.s0 = ((A_q.s3 & (0x0F00)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s7 & (0x0F00)) - 8) * A_d.s1; \
            A.s2 = ((A_q.sB & (0x0F00)) - 8) * A_d.s2; \
            A.s3 = ((A_q.sF & (0x0F00)) - 8) * A_d.s3;

        #define DEQUANTIZE_A_K15 \
            A.s0 = ((A_q.s3 & (0xF000)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s7 & (0xF000)) - 8) * A_d.s1; \
            A.s2 = ((A_q.sB & (0xF000)) - 8) * A_d.s2; \
            A.s3 = ((A_q.sF & (0xF000)) - 8) * A_d.s3;
    #endif

#elif TILE_M == 8
    #define DEFINE_C(n_idx) \
        half8 C##n_idx = 0;

    #if TILE_K == 2
        #define DEQUANTIZE_A_K0 \
            A.s0 = ((A_q.s0 & (0x000F)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s0 & (0x0F00)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s1 & (0x000F)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s1 & (0x0F00)) - 8) * A_d.s3; \
            A.s4 = ((A_q.s2 & (0x000F)) - 8) * A_d.s4; \
            A.s5 = ((A_q.s2 & (0x0F00)) - 8) * A_d.s5; \
            A.s6 = ((A_q.s3 & (0x000F)) - 8) * A_d.s6; \
            A.s7 = ((A_q.s3 & (0x0F00)) - 8) * A_d.s7;

        #define DEQUANTIZE_A_K1 \
            A.s0 = ((A_q.s0 & (0x00F0)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s0 & (0x0F00)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s1 & (0x00F0)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s1 & (0x0F00)) - 8) * A_d.s3; \
            A.s4 = ((A_q.s2 & (0x00F0)) - 8) * A_d.s4; \
            A.s5 = ((A_q.s2 & (0x0F00)) - 8) * A_d.s5; \
            A.s6 = ((A_q.s3 & (0x00F0)) - 8) * A_d.s6; \
            A.s7 = ((A_q.s3 & (0xF000)) - 8) * A_d.s7;

    #elif TILE_K == 4
        #define DEQUANTIZE_A_K0 \
            A.s0 = ((A_q.s0 & (0x000F)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s1 & (0x000F)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s2 & (0x000F)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s3 & (0x000F)) - 8) * A_d.s3; \
            A.s4 = ((A_q.s4 & (0x000F)) - 8) * A_d.s4; \
            A.s5 = ((A_q.s5 & (0x000F)) - 8) * A_d.s5; \
            A.s6 = ((A_q.s6 & (0x000F)) - 8) * A_d.s6; \
            A.s7 = ((A_q.s7 & (0x000F)) - 8) * A_d.s7;

        #define DEQUANTIZE_A_K1 \
            A.s0 = ((A_q.s0 & (0x00F0)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s1 & (0x00F0)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s2 & (0x00F0)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s3 & (0x00F0)) - 8) * A_d.s3; \
            A.s4 = ((A_q.s4 & (0x00F0)) - 8) * A_d.s4; \
            A.s5 = ((A_q.s5 & (0x00F0)) - 8) * A_d.s5; \
            A.s6 = ((A_q.s6 & (0x00F0)) - 8) * A_d.s6; \
            A.s7 = ((A_q.s7 & (0x00F0)) - 8) * A_d.s7;

        #define DEQUANTIZE_A_K2 \
            A.s0 = ((A_q.s0 & (0x0F00)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s1 & (0x0F00)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s2 & (0x0F00)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s3 & (0x0F00)) - 8) * A_d.s3; \
            A.s4 = ((A_q.s4 & (0x0F00)) - 8) * A_d.s4; \
            A.s5 = ((A_q.s5 & (0x0F00)) - 8) * A_d.s5; \
            A.s6 = ((A_q.s6 & (0x0F00)) - 8) * A_d.s6; \
            A.s7 = ((A_q.s7 & (0x0F00)) - 8) * A_d.s7;

        #define DEQUANTIZE_A_K3 \
            A.s0 = ((A_q.s0 & (0xF000)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s1 & (0xF000)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s2 & (0xF000)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s3 & (0xF000)) - 8) * A_d.s3; \
            A.s4 = ((A_q.s4 & (0xF000)) - 8) * A_d.s4; \
            A.s5 = ((A_q.s5 & (0xF000)) - 8) * A_d.s5; \
            A.s6 = ((A_q.s6 & (0xF000)) - 8) * A_d.s6; \
            A.s7 = ((A_q.s7 & (0xF000)) - 8) * A_d.s7;

    #elif TILE_K == 8
        #define DEQUANTIZE_A_K0 \
            A.s0 = ((A_q.s0 & (0x000F)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s2 & (0x000F)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s4 & (0x000F)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s6 & (0x000F)) - 8) * A_d.s3; \
            A.s4 = ((A_q.s8 & (0x000F)) - 8) * A_d.s4; \
            A.s5 = ((A_q.sA & (0x000F)) - 8) * A_d.s5; \
            A.s6 = ((A_q.sC & (0x000F)) - 8) * A_d.s6; \
            A.s7 = ((A_q.sE & (0x000F)) - 8) * A_d.s7;

        #define DEQUANTIZE_A_K1 \
            A.s0 = ((A_q.s0 & (0x00F0)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s2 & (0x00F0)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s4 & (0x00F0)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s6 & (0x00F0)) - 8) * A_d.s3; \
            A.s4 = ((A_q.s8 & (0x00F0)) - 8) * A_d.s4; \
            A.s5 = ((A_q.sA & (0x00F0)) - 8) * A_d.s5; \
            A.s6 = ((A_q.sC & (0x00F0)) - 8) * A_d.s6; \
            A.s7 = ((A_q.sE & (0x00F0)) - 8) * A_d.s7;

        #define DEQUANTIZE_A_K2 \
            A.s0 = ((A_q.s0 & (0x0F00)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s2 & (0x0F00)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s4 & (0x0F00)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s6 & (0x0F00)) - 8) * A_d.s3; \
            A.s4 = ((A_q.s8 & (0x0F00)) - 8) * A_d.s4; \
            A.s5 = ((A_q.sA & (0x0F00)) - 8) * A_d.s5; \
            A.s6 = ((A_q.sC & (0x0F00)) - 8) * A_d.s6; \
            A.s7 = ((A_q.sE & (0x0F00)) - 8) * A_d.s7;

        #define DEQUANTIZE_A_K3 \
            A.s0 = ((A_q.s0 & (0xF000)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s2 & (0xF000)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s4 & (0xF000)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s6 & (0xF000)) - 8) * A_d.s3; \
            A.s4 = ((A_q.s8 & (0xF000)) - 8) * A_d.s4; \
            A.s5 = ((A_q.sA & (0xF000)) - 8) * A_d.s5; \
            A.s6 = ((A_q.sC & (0xF000)) - 8) * A_d.s6; \
            A.s7 = ((A_q.sE & (0xF000)) - 8) * A_d.s7;

        #define DEQUANTIZE_A_K4 \
            A.s0 = ((A_q.s1 & (0x000F)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s3 & (0x000F)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s5 & (0x000F)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s7 & (0x000F)) - 8) * A_d.s3; \
            A.s4 = ((A_q.s9 & (0x000F)) - 8) * A_d.s4; \
            A.s5 = ((A_q.sB & (0x000F)) - 8) * A_d.s5; \
            A.s6 = ((A_q.sD & (0x000F)) - 8) * A_d.s6; \
            A.s7 = ((A_q.sF & (0x000F)) - 8) * A_d.s7;

        #define DEQUANTIZE_A_K5 \
            A.s0 = ((A_q.s1 & (0x00F0)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s3 & (0x00F0)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s5 & (0x00F0)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s7 & (0x00F0)) - 8) * A_d.s3; \
            A.s4 = ((A_q.s9 & (0x00F0)) - 8) * A_d.s4; \
            A.s5 = ((A_q.sB & (0x00F0)) - 8) * A_d.s5; \
            A.s6 = ((A_q.sD & (0x00F0)) - 8) * A_d.s6; \
            A.s7 = ((A_q.sF & (0x00F0)) - 8) * A_d.s7;
            
        #define DEQUANTIZE_A_K6 \
            A.s0 = ((A_q.s1 & (0x0F00)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s3 & (0x0F00)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s5 & (0x0F00)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s7 & (0x0F00)) - 8) * A_d.s3; \
            A.s4 = ((A_q.s9 & (0x0F00)) - 8) * A_d.s4; \
            A.s5 = ((A_q.sB & (0x0F00)) - 8) * A_d.s5; \
            A.s6 = ((A_q.sD & (0x0F00)) - 8) * A_d.s6; \
            A.s7 = ((A_q.sF & (0x0F00)) - 8) * A_d.s7;

        #define DEQUANTIZE_A_K7 \
            A.s0 = ((A_q.s1 & (0xF000)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s3 & (0xF000)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s5 & (0xF000)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s7 & (0xF000)) - 8) * A_d.s3; \
            A.s4 = ((A_q.s9 & (0xF000)) - 8) * A_d.s4; \
            A.s5 = ((A_q.sB & (0xF000)) - 8) * A_d.s5; \
            A.s6 = ((A_q.sD & (0xF000)) - 8) * A_d.s6; \
            A.s7 = ((A_q.sF & (0xF000)) - 8) * A_d.s7;
    #endif

#elif TILE_M == 16
    #define DEFINE_C(n_idx) \
        half16 C##n_idx = 0;

    #if TILE_K == 1
        #define DEQUANTIZE_A_K0 \
            A.s0 = ((A_q.s0 & (0x000F)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s0 & (0x00F0)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s0 & (0x0F00)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s0 & (0xF000)) - 8) * A_d.s3; \
            A.s4 = ((A_q.s1 & (0x000F)) - 8) * A_d.s4; \
            A.s5 = ((A_q.s1 & (0x00F0)) - 8) * A_d.s5; \
            A.s6 = ((A_q.s1 & (0x0F00)) - 8) * A_d.s6; \
            A.s7 = ((A_q.s1 & (0xF000)) - 8) * A_d.s7; \
            A.s8 = ((A_q.s2 & (0x000F)) - 8) * A_d.s8; \
            A.s9 = ((A_q.s2 & (0x00F0)) - 8) * A_d.s9; \
            A.sA = ((A_q.s2 & (0x0F00)) - 8) * A_d.sA; \
            A.sB = ((A_q.s2 & (0xF000)) - 8) * A_d.sB; \
            A.sC = ((A_q.s3 & (0x000F)) - 8) * A_d.sC; \
            A.sD = ((A_q.s3 & (0x00F0)) - 8) * A_d.sD; \
            A.sE = ((A_q.s3 & (0x0F00)) - 8) * A_d.sE; \
            A.sF = ((A_q.s3 & (0xF000)) - 8) * A_d.sF;

    #elif TILE_K == 2
        #define DEQUANTIZE_A_K0 \
            A.s0 = ((A_q.s0 & (0x000F)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s0 & (0x0F00)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s1 & (0x000F)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s1 & (0x0F00)) - 8) * A_d.s3; \
            A.s4 = ((A_q.s2 & (0x000F)) - 8) * A_d.s4; \
            A.s5 = ((A_q.s2 & (0x0F00)) - 8) * A_d.s5; \
            A.s6 = ((A_q.s3 & (0x000F)) - 8) * A_d.s6; \
            A.s7 = ((A_q.s3 & (0x0F00)) - 8) * A_d.s7; \
            A.s8 = ((A_q.s4 & (0x000F)) - 8) * A_d.s8; \
            A.s9 = ((A_q.s4 & (0x0F00)) - 8) * A_d.s9; \
            A.sA = ((A_q.s5 & (0x000F)) - 8) * A_d.sA; \
            A.sB = ((A_q.s5 & (0x0F00)) - 8) * A_d.sB; \
            A.sC = ((A_q.s6 & (0x000F)) - 8) * A_d.sC; \
            A.sD = ((A_q.s6 & (0x0F00)) - 8) * A_d.sD; \
            A.sE = ((A_q.s7 & (0x000F)) - 8) * A_d.sE; \
            A.sF = ((A_q.s7 & (0x0F00)) - 8) * A_d.sF;

        #define DEQUANTIZE_A_K1 \
            A.s0 = ((A_q.s0 & (0x00F0)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s0 & (0xF000)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s1 & (0x00F0)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s1 & (0xF000)) - 8) * A_d.s3; \
            A.s4 = ((A_q.s2 & (0x00F0)) - 8) * A_d.s4; \
            A.s5 = ((A_q.s2 & (0xF000)) - 8) * A_d.s5; \
            A.s6 = ((A_q.s3 & (0x00F0)) - 8) * A_d.s6; \
            A.s7 = ((A_q.s3 & (0xF000)) - 8) * A_d.s7; \
            A.s8 = ((A_q.s4 & (0x00F0)) - 8) * A_d.s8; \
            A.s9 = ((A_q.s4 & (0xF000)) - 8) * A_d.s9; \
            A.sA = ((A_q.s5 & (0x00F0)) - 8) * A_d.sA; \
            A.sB = ((A_q.s5 & (0xF000)) - 8) * A_d.sB; \
            A.sC = ((A_q.s6 & (0x00F0)) - 8) * A_d.sC; \
            A.sD = ((A_q.s6 & (0xF000)) - 8) * A_d.sD; \
            A.sE = ((A_q.s7 & (0x00F0)) - 8) * A_d.sE; \
            A.sF = ((A_q.s7 & (0xF000)) - 8) * A_d.sF;

    #elif TILE_K == 4
        #define DEQUANTIZE_A_K0 \
            A.s0 = ((A_q.s0 & (0x000F)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s1 & (0x000F)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s2 & (0x000F)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s3 & (0x000F)) - 8) * A_d.s3; \
            A.s4 = ((A_q.s4 & (0x000F)) - 8) * A_d.s4; \
            A.s5 = ((A_q.s5 & (0x000F)) - 8) * A_d.s5; \
            A.s6 = ((A_q.s6 & (0x000F)) - 8) * A_d.s6; \
            A.s7 = ((A_q.s7 & (0x000F)) - 8) * A_d.s7; \
            A.s8 = ((A_q.s8 & (0x000F)) - 8) * A_d.s8; \
            A.s9 = ((A_q.s9 & (0x000F)) - 8) * A_d.s9; \
            A.sA = ((A_q.sA & (0x000F)) - 8) * A_d.sA; \
            A.sB = ((A_q.sB & (0x000F)) - 8) * A_d.sB; \
            A.sC = ((A_q.sC & (0x000F)) - 8) * A_d.sC; \
            A.sD = ((A_q.sD & (0x000F)) - 8) * A_d.sD; \
            A.sE = ((A_q.sE & (0x000F)) - 8) * A_d.sE; \
            A.sF = ((A_q.sF & (0x000F)) - 8) * A_d.sF;

        #define DEQUANTIZE_A_K1 \
            A.s0 = ((A_q.s0 & (0x00F0)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s1 & (0x00F0)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s2 & (0x00F0)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s3 & (0x00F0)) - 8) * A_d.s3; \
            A.s4 = ((A_q.s4 & (0x00F0)) - 8) * A_d.s4; \
            A.s5 = ((A_q.s5 & (0x00F0)) - 8) * A_d.s5; \
            A.s6 = ((A_q.s6 & (0x00F0)) - 8) * A_d.s6; \
            A.s7 = ((A_q.s7 & (0x00F0)) - 8) * A_d.s7; \
            A.s8 = ((A_q.s8 & (0x00F0)) - 8) * A_d.s8; \
            A.s9 = ((A_q.s9 & (0x00F0)) - 8) * A_d.s9; \
            A.sA = ((A_q.sA & (0x00F0)) - 8) * A_d.sA; \
            A.sB = ((A_q.sB & (0x00F0)) - 8) * A_d.sB; \
            A.sC = ((A_q.sC & (0x00F0)) - 8) * A_d.sC; \
            A.sD = ((A_q.sD & (0x00F0)) - 8) * A_d.sD; \
            A.sE = ((A_q.sE & (0x00F0)) - 8) * A_d.sE; \
            A.sF = ((A_q.sF & (0x00F0)) - 8) * A_d.sF;

        #define DEQUANTIZE_A_K2 \
            A.s0 = ((A_q.s0 & (0x0F00)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s1 & (0x0F00)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s2 & (0x0F00)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s3 & (0x0F00)) - 8) * A_d.s3; \
            A.s4 = ((A_q.s4 & (0x0F00)) - 8) * A_d.s4; \
            A.s5 = ((A_q.s5 & (0x0F00)) - 8) * A_d.s5; \
            A.s6 = ((A_q.s6 & (0x0F00)) - 8) * A_d.s6; \
            A.s7 = ((A_q.s7 & (0x0F00)) - 8) * A_d.s7; \
            A.s8 = ((A_q.s8 & (0x0F00)) - 8) * A_d.s8; \
            A.s9 = ((A_q.s9 & (0x0F00)) - 8) * A_d.s9; \
            A.sA = ((A_q.sA & (0x0F00)) - 8) * A_d.sA; \
            A.sB = ((A_q.sB & (0x0F00)) - 8) * A_d.sB; \
            A.sC = ((A_q.sC & (0x0F00)) - 8) * A_d.sC; \
            A.sD = ((A_q.sD & (0x0F00)) - 8) * A_d.sD; \
            A.sE = ((A_q.sE & (0x0F00)) - 8) * A_d.sE; \
            A.sF = ((A_q.sF & (0x0F00)) - 8) * A_d.sF;

        #define DEQUANTIZE_A_K3 \
            A.s0 = ((A_q.s0 & (0xF000)) - 8) * A_d.s0; \
            A.s1 = ((A_q.s1 & (0xF000)) - 8) * A_d.s1; \
            A.s2 = ((A_q.s2 & (0xF000)) - 8) * A_d.s2; \
            A.s3 = ((A_q.s3 & (0xF000)) - 8) * A_d.s3; \
            A.s4 = ((A_q.s4 & (0xF000)) - 8) * A_d.s4; \
            A.s5 = ((A_q.s5 & (0xF000)) - 8) * A_d.s5; \
            A.s6 = ((A_q.s6 & (0xF000)) - 8) * A_d.s6; \
            A.s7 = ((A_q.s7 & (0xF000)) - 8) * A_d.s7; \
            A.s8 = ((A_q.s8 & (0xF000)) - 8) * A_d.s8; \
            A.s9 = ((A_q.s9 & (0xF000)) - 8) * A_d.s9; \
            A.sA = ((A_q.sA & (0xF000)) - 8) * A_d.sA; \
            A.sB = ((A_q.sB & (0xF000)) - 8) * A_d.sB; \
            A.sC = ((A_q.sC & (0xF000)) - 8) * A_d.sC; \
            A.sD = ((A_q.sD & (0xF000)) - 8) * A_d.sD; \
            A.sE = ((A_q.sE & (0xF000)) - 8) * A_d.sE; \
            A.sF = ((A_q.sF & (0xF000)) - 8) * A_d.sF;
    #endif
#endif

#if TILE_K == 1
    #define DEFINE_B(n_idx) \
        half B##n_idx = 0;

    #define LOAD_B(n_idx) \
        B##n_idx = read_imageh(src1, B_idx).s0;
    
#elif TILE_K == 2
    #define DEFINE_B(n_idx) \
        half2 B##n_idx = 0;

    #define LOAD_B(n_idx) \
        B##n_idx = read_imageh(src1, B_idx).s01;

#elif TILE_K == 4
    #define DEFINE_B(n_idx) \
        half4 B##n_idx = 0;

    #define LOAD_B(n_idx) \
        B##n_idx = read_imageh(src1, B_idx);

#elif TILE_K == 8
    #define DEFINE_B(n_idx) \
        half8 B##n_idx = 0;

    #define LOAD_B(n_idx) \
        B##n_idx.s0123 = read_imageh(src1, B_idx); \
        B##n_idx.s4567 = read_imageh(src1, B_idx + 1);

#elif TILE_K == 16
    #define DEFINE_B(n_idx) \
        half16 B##n_idx = 0;

    #define LOAD_B(n_idx) \
        B##n_idx.s0123 = read_imageh(src1, B_idx); \
        B##n_idx.s4567 = read_imageh(src1, B_idx + 1); \
        B##n_idx.s89ab = read_imageh(src1, B_idx + 2); \
        B##n_idx.s0123 = read_imageh(src1, B_idx + 3);
#endif


#if TILE_N == 1
    #if TILE_K == 1
        #define COMPUTE \
            DEQUANTIZE_A_K0; \
            C0 += B0 * A;

    #elif TILE_K == 2
        #define COMPUTE \
            DEQUANTIZE_A_K0; \
            C0 += B0.s0 * A; \
            DEQUANTIZE_A_K1; \
            C0 += B0.s1 * A;

    #elif TILE_K == 4
        #define COMPUTE \
            DEQUANTIZE_A_K0; \
            C0 += B0.s0 * A; \
            DEQUANTIZE_A_K1; \
            C0 += B0.s1 * A; \
            DEQUANTIZE_A_K2; \
            C0 += B0.s2 * A; \
            DEQUANTIZE_A_K3; \
            C0 += B0.s3 * A;

    #elif TILE_K == 8
        #define COMPUTE \
            DEQUANTIZE_A_K0; \
            C0 += B0.s0 * A; \
            DEQUANTIZE_A_K1; \
            C0 += B0.s1 * A; \
            DEQUANTIZE_A_K2; \
            C0 += B0.s2 * A; \
            DEQUANTIZE_A_K3; \
            C0 += B0.s3 * A; \
            DEQUANTIZE_A_K4; \
            C0 += B0.s4 * A; \
            DEQUANTIZE_A_K5; \
            C0 += B0.s5 * A; \
            DEQUANTIZE_A_K6; \
            C0 += B0.s6 * A; \
            DEQUANTIZE_A_K7; \
            C0 += B0.s7 * A;

    #elif TILE_K == 16
        #define COMPUTE \
            DEQUANTIZE_A_K0; \
            C0 += B0.s0 * A; \
            DEQUANTIZE_A_K1; \
            C0 += B0.s1 * A; \
            DEQUANTIZE_A_K2; \
            C0 += B0.s2 * A; \
            DEQUANTIZE_A_K3; \
            C0 += B0.s3 * A; \
            DEQUANTIZE_A_K4; \
            C0 += B0.s4 * A; \
            DEQUANTIZE_A_K5; \
            C0 += B0.s5 * A; \
            DEQUANTIZE_A_K6; \
            C0 += B0.s6 * A; \
            DEQUANTIZE_A_K7; \
            C0 += B0.s7 * A; \
            DEQUANTIZE_A_K8; \
            C0 += B0.s8 * A; \
            DEQUANTIZE_A_K9; \
            C0 += B0.s9 * A; \
            DEQUANTIZE_A_K10; \
            C0 += B0.sA * A; \
            DEQUANTIZE_A_K11; \
            C0 += B0.sB * A; \
            DEQUANTIZE_A_K12; \
            C0 += B0.sC * A; \
            DEQUANTIZE_A_K13; \
            C0 += B0.sD * A; \
            DEQUANTIZE_A_K14; \
            C0 += B0.sE * A; \
            DEQUANTIZE_A_K15; \
            C0 += B0.sF * A;
    #endif

#elif TILE_N == 2
    #if TILE_K == 1
        #define COMPUTE() \
            DEQUANTIZE_A_K0; \
            C0 += B0 * A; \
            C1 += B1 * A;
    #elif TILE_K == 2
        #define COMPUTE() \
            DEQUANTIZE_A_K0; \
            C0 += B0.s0 * A; \
            C1 += B1.s0 * A; \
            DEQUANTIZE_A_K1; \
            C0 += B0.s1 * A; \
            C1 += B1.s1 * A;
    #elif TILE_K == 4
        #define COMPUTE() \
            DEQUANTIZE_A_K0; \
            C0 += B0.s0 * A; \
            C1 += B1.s0 * A; \
            DEQUANTIZE_A_K1; \
            C0 += B0.s1 * A; \
            C1 += B1.s1 * A; \
            DEQUANTIZE_A_K2; \
            C0 += B0.s2 * A; \
            C1 += B1.s2 * A; \
            DEQUANTIZE_A_K2; \
            C0 += B0.s3 * A; \
            C1 += B1.s3 * A;
    #elif TILE_K == 8
        #define COMPUTE() \
            DEQUANTIZE_A_K0; \
            C0 += B0.s0 * A; \
            C1 += B1.s0 * A; \
            DEQUANTIZE_A_K1; \
            C0 += B0.s1 * A; \
            C1 += B1.s1 * A; \
            DEQUANTIZE_A_K2; \
            C0 += B0.s2 * A; \
            C1 += B1.s2 * A; \
            DEQUANTIZE_A_K3; \
            C0 += B0.s3 * A; \
            C1 += B1.s3 * A; \
            DEQUANTIZE_A_K4; \
            C0 += B0.s4 * A; \
            C1 += B1.s4 * A; \
            DEQUANTIZE_A_K5; \
            C0 += B0.s5 * A; \
            C1 += B1.s5 * A; \
            DEQUANTIZE_A_K6; \
            C0 += B0.s6 * A; \
            C1 += B1.s6 * A; \
            DEQUANTIZE_A_K7; \
            C0 += B0.s7 * A; \
            C1 += B1.s7 * A;
    #elif TILE_K == 16
        #define COMPUTE() \
            DEQUANTIZE_A_K0; \
            C0 += B0.s0 * A; \
            C1 += B1.s0 * A; \
            DEQUANTIZE_A_K1; \
            C0 += B0.s1 * A; \
            C1 += B1.s1 * A; \
            DEQUANTIZE_A_K2; \
            C0 += B0.s2 * A; \
            C1 += B1.s2 * A; \
            DEQUANTIZE_A_K3; \
            C0 += B0.s3 * A; \
            C1 += B1.s3 * A; \
            DEQUANTIZE_A_K4; \
            C0 += B0.s4 * A; \
            C1 += B1.s4 * A; \
            DEQUANTIZE_A_K5; \
            C0 += B0.s5 * A; \
            C1 += B1.s5 * A; \
            DEQUANTIZE_A_K6; \
            C0 += B0.s6 * A; \
            C1 += B1.s6 * A; \
            DEQUANTIZE_A_K7; \
            C0 += B0.s7 * A; \
            C1 += B1.s7 * A; \
            DEQUANTIZE_A_K8; \
            C0 += B0.s8 * A; \
            C1 += B1.s8 * A; \
            DEQUANTIZE_A_K9; \
            C0 += B0.s9 * A; \
            C1 += B1.s9 * A; \
            DEQUANTIZE_A_K10; \
            C0 += B0.sA * A; \
            C1 += B1.sA * A; \
            DEQUANTIZE_A_K11; \
            C0 += B0.sB * A; \
            C1 += B1.sB * A; \
            DEQUANTIZE_A_K12; \
            C0 += B0.sC * A; \
            C1 += B1.sC * A; \
            DEQUANTIZE_A_K13; \
            C0 += B0.sD * A; \
            C1 += B1.sD * A; \
            DEQUANTIZE_A_K14; \
            C0 += B0.sE * A; \
            C1 += B1.sE * A; \
            DEQUANTIZE_A_K15; \
            C0 += B0.sF * A; \
            C1 += B1.sF * A;
    #endif
#elif TILE_N == 4
    #if TILE_K == 1
        #define COMPUTE() \
            DEQUANTIZE_A_K0; \
            C0 += B0 * A; \
            C1 += B1 * A; \
            C2 += B2 * A; \
            C3 += B3 * A;
    #elif TILE_K == 2
        #define COMPUTE() \
            DEQUANTIZE_A_K0; \
            C0 += B0.s0 * A; \
            C1 += B1.s0 * A; \
            C2 += B2.s0 * A; \
            C3 += B3.s0 * A; \
            DEQUANTIZE_A_K1; \
            C0 += B0.s1 * A; \
            C1 += B1.s1 * A; \
            C2 += B2.s1 * A; \
            C3 += B3.s1 * A;
    #elif TILE_K == 4
        #define COMPUTE() \
            DEQUANTIZE_A_K0; \
            C0 += B0.s0 * A; \
            C1 += B1.s0 * A; \
            C2 += B2.s0 * A; \
            C3 += B3.s0 * A; \
            DEQUANTIZE_A_K1; \
            C0 += B0.s1 * A; \
            C1 += B1.s1 * A; \
            C2 += B2.s1 * A; \
            C3 += B3.s1 * A; \
            DEQUANTIZE_A_K2; \
            C0 += B0.s2 * A; \
            C1 += B1.s2 * A; \
            C2 += B2.s2 * A; \
            C3 += B3.s2 * A; \
            DEQUANTIZE_A_K3; \
            C0 += B0.s3 * A; \
            C1 += B1.s3 * A; \
            C2 += B2.s3 * A; \
            C3 += B3.s3 * A;
    #elif TILE_K == 8
        #define COMPUTE() \
            DEQUANTIZE_A_K0; \
            C0 += B0.s0 * A; \
            C1 += B1.s0 * A; \
            C2 += B2.s0 * A; \
            C3 += B3.s0 * A; \
            DEQUANTIZE_A_K1; \
            C0 += B0.s1 * A; \
            C1 += B1.s1 * A; \
            C2 += B2.s1 * A; \
            C3 += B3.s1 * A; \
            DEQUANTIZE_A_K2; \
            C0 += B0.s2 * A; \
            C1 += B1.s2 * A; \
            C2 += B2.s2 * A; \
            C3 += B3.s2 * A; \
            DEQUANTIZE_A_K3; \
            C0 += B0.s3 * A; \
            C1 += B1.s3 * A; \
            C2 += B2.s3 * A; \
            C3 += B3.s3 * A; \
            DEQUANTIZE_A_K4; \
            C0 += B0.s4 * A; \
            C1 += B1.s4 * A; \
            C2 += B2.s4 * A; \
            C3 += B3.s4 * A; \
            DEQUANTIZE_A_K5; \
            C0 += B0.s5 * A; \
            C1 += B1.s5 * A; \
            C2 += B2.s5 * A; \
            C3 += B3.s5 * A; \
            DEQUANTIZE_A_K6; \
            C0 += B0.s6 * A; \
            C1 += B1.s6 * A; \
            C2 += B2.s6 * A; \
            C3 += B3.s6 * A; \
            DEQUANTIZE_A_K7; \
            C0 += B0.s7 * A; \
            C1 += B1.s7 * A; \
            C2 += B2.s7 * A; \
            C3 += B3.s7 * A;
    #elif TILE_K == 16
        #define COMPUTE() \
            DEQUANTIZE_A_K0; \
            C0 += B0.s0 * A; \
            C1 += B1.s0 * A; \
            C2 += B2.s0 * A; \
            C3 += B3.s0 * A; \
            DEQUANTIZE_A_K1; \
            C0 += B0.s1 * A; \
            C1 += B1.s1 * A; \
            C2 += B2.s1 * A; \
            C3 += B3.s1 * A; \
            DEQUANTIZE_A_K2; \
            C0 += B0.s2 * A; \
            C1 += B1.s2 * A; \
            C2 += B2.s2 * A; \
            C3 += B3.s2 * A; \
            DEQUANTIZE_A_K3; \
            C0 += B0.s3 * A; \
            C1 += B1.s3 * A; \
            C2 += B2.s3 * A; \
            C3 += B3.s3 * A; \
            DEQUANTIZE_A_K4; \
            C0 += B0.s4 * A; \
            C1 += B1.s4 * A; \
            C2 += B2.s4 * A; \
            C3 += B3.s4 * A; \
            DEQUANTIZE_A_K5; \
            C0 += B0.s5 * A; \
            C1 += B1.s5 * A; \
            C2 += B2.s5 * A; \
            C3 += B3.s5 * A; \
            DEQUANTIZE_A_K6; \
            C0 += B0.s6 * A; \
            C1 += B1.s6 * A; \
            C2 += B2.s6 * A; \
            C3 += B3.s6 * A; \
            DEQUANTIZE_A_K7; \
            C0 += B0.s7 * A; \
            C1 += B1.s7 * A; \
            C2 += B2.s7 * A; \
            C3 += B3.s7 * A; \
            DEQUANTIZE_A_K8; \
            C0 += B0.s8 * A; \
            C1 += B1.s8 * A; \
            C2 += B2.s8 * A; \
            C3 += B3.s8 * A; \
            DEQUANTIZE_A_K9; \
            C0 += B0.s9 * A; \
            C1 += B1.s9 * A; \
            C2 += B2.s9 * A; \
            C3 += B3.s9 * A; \
            DEQUANTIZE_A_K10; \
            C0 += B0.sA * A; \
            C1 += B1.sA * A; \
            C2 += B2.sA * A; \
            C3 += B3.sA * A; \
            DEQUANTIZE_A_K11; \
            C0 += B0.sB * A; \
            C1 += B1.sB * A; \
            C2 += B2.sB * A; \
            C3 += B3.sB * A; \
            DEQUANTIZE_A_K12; \
            C0 += B0.sC * A; \
            C1 += B1.sC * A; \
            C2 += B2.sC * A; \
            C3 += B3.sC * A; \
            DEQUANTIZE_A_K13; \
            C0 += B0.sD * A; \
            C1 += B1.sD * A; \
            C2 += B2.sD * A; \
            C3 += B3.sD * A; \
            DEQUANTIZE_A_K14; \
            C0 += B0.sE * A; \
            C1 += B1.sE * A; \
            C2 += B2.sE * A; \
            C3 += B3.sE * A; \
            DEQUANTIZE_A_K15; \
            C0 += B0.sF * A; \
            C1 += B1.sF * A; \
            C2 += B2.sF * A; \
            C3 += B3.sF * A;
    #endif
#elif TILE_N == 8
    #if TILE_K == 1
        #define COMPUTE() \
            DEQUANTIZE_A_K0; \
            C0 += B0 * A; \
            C1 += B1 * A; \
            C2 += B2 * A; \
            C3 += B3 * A; \
            C4 += B4 * A; \
            C5 += B5 * A; \
            C6 += B6 * A; \
            C7 += B7 * A;
    #elif TILE_K == 2
        #define COMPUTE() \
            DEQUANTIZE_A_K0; \
            C0 += B0.s0 * A; \
            C1 += B1.s0 * A; \
            C2 += B2.s0 * A; \
            C3 += B3.s0 * A; \
            C4 += B4.s0 * A; \
            C5 += B5.s0 * A; \
            C6 += B6.s0 * A; \
            C7 += B7.s0 * A; \
            DEQUANTIZE_A_K1; \
            C0 += B0.s1 * A; \
            C1 += B1.s1 * A; \
            C2 += B2.s1 * A; \
            C3 += B3.s1 * A; \
            C4 += B4.s1 * A; \
            C5 += B5.s1 * A; \
            C6 += B6.s1 * A; \
            C7 += B7.s1 * A;
    #elif TILE_K == 4
        #define COMPUTE() \
            DEQUANTIZE_A_K0; \
            C0 += B0.s0 * A; \
            C1 += B1.s0 * A; \
            C2 += B2.s0 * A; \
            C3 += B3.s0 * A; \
            C4 += B4.s0 * A; \
            C5 += B5.s0 * A; \
            C6 += B6.s0 * A; \
            C7 += B7.s0 * A; \
            DEQUANTIZE_A_K1; \
            C0 += B0.s1 * A; \
            C1 += B1.s1 * A; \
            C2 += B2.s1 * A; \
            C3 += B3.s1 * A; \
            C4 += B4.s1 * A; \
            C5 += B5.s1 * A; \
            C6 += B6.s1 * A; \
            C7 += B7.s1 * A; \
            DEQUANTIZE_A_K2; \
            C0 += B0.s2 * A; \
            C1 += B1.s2 * A; \
            C2 += B2.s2 * A; \
            C3 += B3.s2 * A; \
            C4 += B4.s2 * A; \
            C5 += B5.s2 * A; \
            C6 += B6.s2 * A; \
            C7 += B7.s2 * A; \
            DEQUANTIZE_A_K3; \
            C0 += B0.s3 * A; \
            C1 += B1.s3 * A; \
            C2 += B2.s3 * A; \
            C3 += B3.s3 * A; \
            C4 += B4.s3 * A; \
            C5 += B5.s3 * A; \
            C6 += B6.s3 * A; \
            C7 += B7.s3 * A;
    #elif TILE_K == 8
        #define COMPUTE() \
            DEQUANTIZE_A_K0; \
            C0 += B0.s0 * A; \
            C1 += B1.s0 * A; \
            C2 += B2.s0 * A; \
            C3 += B3.s0 * A; \
            C4 += B4.s0 * A; \
            C5 += B5.s0 * A; \
            C6 += B6.s0 * A; \
            C7 += B7.s0 * A; \
            DEQUANTIZE_A_K1; \
            C0 += B0.s1 * A; \
            C1 += B1.s1 * A; \
            C2 += B2.s1 * A; \
            C3 += B3.s1 * A; \
            C4 += B4.s1 * A; \
            C5 += B5.s1 * A; \
            C6 += B6.s1 * A; \
            C7 += B7.s1 * A; \
            DEQUANTIZE_A_K2; \
            C0 += B0.s2 * A; \
            C1 += B1.s2 * A; \
            C2 += B2.s2 * A; \
            C3 += B3.s2 * A; \
            C4 += B4.s2 * A; \
            C5 += B5.s2 * A; \
            C6 += B6.s2 * A; \
            C7 += B7.s2 * A; \
            DEQUANTIZE_A_K3; \
            C0 += B0.s3 * A; \
            C1 += B1.s3 * A; \
            C2 += B2.s3 * A; \
            C3 += B3.s3 * A; \
            C4 += B4.s3 * A; \
            C5 += B5.s3 * A; \
            C6 += B6.s3 * A; \
            C7 += B7.s3 * A; \
            DEQUANTIZE_A_K4; \
            C0 += B0.s4 * A; \
            C1 += B1.s4 * A; \
            C2 += B2.s4 * A; \
            C3 += B3.s4 * A; \
            C4 += B4.s4 * A; \
            C5 += B5.s4 * A; \
            C6 += B6.s4 * A; \
            C7 += B7.s4 * A; \
            DEQUANTIZE_A_K5; \
            C0 += B0.s5 * A; \
            C1 += B1.s5 * A; \
            C2 += B2.s5 * A; \
            C3 += B3.s5 * A; \
            C4 += B4.s5 * A; \
            C5 += B5.s5 * A; \
            C6 += B6.s5 * A; \
            C7 += B7.s5 * A; \
            DEQUANTIZE_A_K6; \
            C0 += B0.s6 * A; \
            C1 += B1.s6 * A; \
            C2 += B2.s6 * A; \
            C3 += B3.s6 * A; \
            C4 += B4.s6 * A; \
            C5 += B5.s6 * A; \
            C6 += B6.s6 * A; \
            C7 += B7.s6 * A; \
            DEQUANTIZE_A_K7; \
            C0 += B0.s7 * A; \
            C1 += B1.s7 * A; \
            C2 += B2.s7 * A; \
            C3 += B3.s7 * A; \
            C4 += B4.s7 * A; \
            C5 += B5.s7 * A; \
            C6 += B6.s7 * A; \
            C7 += B7.s7 * A;
    #elif TILE_K == 16
        #define COMPUTE() \
            DEQUANTIZE_A_K0; \
            C0 += B0.s0 * A; \
            C1 += B1.s0 * A; \
            C2 += B2.s0 * A; \
            C3 += B3.s0 * A; \
            C4 += B4.s0 * A; \
            C5 += B5.s0 * A; \
            C6 += B6.s0 * A; \
            C7 += B7.s0 * A; \
            DEQUANTIZE_A_K1; \
            C0 += B0.s1 * A; \
            C1 += B1.s1 * A; \
            C2 += B2.s1 * A; \
            C3 += B3.s1 * A; \
            C4 += B4.s1 * A; \
            C5 += B5.s1 * A; \
            C6 += B6.s1 * A; \
            C7 += B7.s1 * A; \
            DEQUANTIZE_A_K2; \
            C0 += B0.s2 * A; \
            C1 += B1.s2 * A; \
            C2 += B2.s2 * A; \
            C3 += B3.s2 * A; \
            C4 += B4.s2 * A; \
            C5 += B5.s2 * A; \
            C6 += B6.s2 * A; \
            C7 += B7.s2 * A; \
            DEQUANTIZE_A_K3; \
            C0 += B0.s3 * A; \
            C1 += B1.s3 * A; \
            C2 += B2.s3 * A; \
            C3 += B3.s3 * A; \
            C4 += B4.s3 * A; \
            C5 += B5.s3 * A; \
            C6 += B6.s3 * A; \
            C7 += B7.s3 * A; \
            DEQUANTIZE_A_K4; \
            C0 += B0.s4 * A; \
            C1 += B1.s4 * A; \
            C2 += B2.s4 * A; \
            C3 += B3.s4 * A; \
            C4 += B4.s4 * A; \
            C5 += B5.s4 * A; \
            C6 += B6.s4 * A; \
            C7 += B7.s4 * A; \
            DEQUANTIZE_A_K5; \
            C0 += B0.s5 * A; \
            C1 += B1.s5 * A; \
            C2 += B2.s5 * A; \
            C3 += B3.s5 * A; \
            C4 += B4.s5 * A; \
            C5 += B5.s5 * A; \
            C6 += B6.s5 * A; \
            C7 += B7.s5 * A; \
            DEQUANTIZE_A_K6; \
            C0 += B0.s6 * A; \
            C1 += B1.s6 * A; \
            C2 += B2.s6 * A; \
            C3 += B3.s6 * A; \
            C4 += B4.s6 * A; \
            C5 += B5.s6 * A; \
            C6 += B6.s6 * A; \
            C7 += B7.s6 * A; \
            DEQUANTIZE_A_K7; \
            C0 += B0.s7 * A; \
            C1 += B1.s7 * A; \
            C2 += B2.s7 * A; \
            C3 += B3.s7 * A; \
            C4 += B4.s7 * A; \
            C5 += B5.s7 * A; \
            C6 += B6.s7 * A; \
            C7 += B7.s7 * A; \
            DEQUANTIZE_A_K8; \
            C0 += B0.s8 * A; \
            C1 += B1.s8 * A; \
            C2 += B2.s8 * A; \
            C3 += B3.s8 * A; \
            C4 += B4.s8 * A; \
            C5 += B5.s8 * A; \
            C6 += B6.s8 * A; \
            C7 += B7.s8 * A; \
            DEQUANTIZE_A_K9; \
            C0 += B0.s9 * A; \
            C1 += B1.s9 * A; \
            C2 += B2.s9 * A; \
            C3 += B3.s9 * A; \
            C4 += B4.s9 * A; \
            C5 += B5.s9 * A; \
            C6 += B6.s9 * A; \
            C7 += B7.s9 * A; \
            DEQUANTIZE_A_K10; \
            C0 += B0.sA * A; \
            C1 += B1.sA * A; \
            C2 += B2.sA * A; \
            C3 += B3.sA * A; \
            C4 += B4.sA * A; \
            C5 += B5.sA * A; \
            C6 += B6.sA * A; \
            C7 += B7.sA * A; \
            DEQUANTIZE_A_K11; \
            C0 += B0.sB * A; \
            C1 += B1.sB * A; \
            C2 += B2.sB * A; \
            C3 += B3.sB * A; \
            C4 += B4.sB * A; \
            C5 += B5.sB * A; \
            C6 += B6.sB * A; \
            C7 += B7.sB * A; \
            DEQUANTIZE_A_K12; \
            C0 += B0.sC * A; \
            C1 += B1.sC * A; \
            C2 += B2.sC * A; \
            C3 += B3.sC * A; \
            C4 += B4.sC * A; \
            C5 += B5.sC * A; \
            C6 += B6.sC * A; \
            C7 += B7.sC * A; \
            DEQUANTIZE_A_K13; \
            C0 += B0.sD * A; \
            C1 += B1.sD * A; \
            C2 += B2.sD * A; \
            C3 += B3.sD * A; \
            C4 += B4.sD * A; \
            C5 += B5.sD * A; \
            C6 += B6.sD * A; \
            C7 += B7.sD * A; \
            DEQUANTIZE_A_K14; \
            C0 += B0.sE * A; \
            C1 += B1.sE * A; \
            C2 += B2.sE * A; \
            C3 += B3.sE * A; \
            C4 += B4.sE * A; \
            C5 += B5.sE * A; \
            C6 += B6.sE * A; \
            C7 += B7.sE * A; \
            DEQUANTIZE_A_K15; \
            C0 += B0.sF * A; \
            C1 += B1.sF * A; \
            C2 += B2.sF * A; \
            C3 += B3.sF * A; \
            C4 += B4.sF * A; \
            C5 += B5.sF * A; \
            C6 += B6.sF * A;
    #endif
#endif


kernel void kernel_mul_mat_q4_0_f32(
        global const ushort * src0_q,       // quantized A
        global const half  * src0_d,        // A scales
        __read_only image1d_buffer_t src1,  // B (1d image)
        global float * dst,                 // C
        int m,                              // M
        int n,                              // N with padding
        int k,                              // K
        int n_no_padding                    // N without padding
) {
#if TILE_N_x_GLOBAL_N_SIZE < N
    for (int n_idx = GLOBAL_N_IDX * TILE_N; n_idx < n; n_idx += TILE_N * GLOBAL_N_SIZE) {
#else
    int n_idx = GLOBAL_N_IDX * TILE_N; {
#endif
#if TILE_M_x_GLOBAL_M_SIZE < M
        // for (int m_idx = GLOBAL_M_IDX * TILE_M; m_idx < m; m_idx += TILE_M * GLOBAL_M_SIZE) {
        for (int m_iter = 0; m_iter < m / (GLOBAL_M_SIZE * TILE_M); m_iter++) {
            int m_idx = (m_iter * GLOBAL_M_SIZE + GLOBAL_M_IDX) * TILE_M;
#else
        int m_iter = 0;
        int m_idx = GLOBAL_M_IDX * TILE_M; {
#endif
            #if TILE_N == 1
                DEFINE_C(0);
            #endif
            #if TILE_N >= 2
                DEFINE_C(1);
            #endif
            #if TILE_N >= 4
                DEFINE_C(2);
                DEFINE_C(3);
            #endif
            #if TILE_N >= 8
                DEFINE_C(4);
                DEFINE_C(5);
                DEFINE_C(6);
                DEFINE_C(7);
            #endif

            // for (int k_idx = GLOBAL_K_IDX * TILE_K; k_idx < k; k_idx += TILE_K * GLOBAL_K_SIZE) {
            for (int k_iter = 0; k_iter < k / TILE_K; k_iter++) {
                int k_idx = (k_iter * GLOBAL_K_SIZE + GLOBAL_K_IDX) * TILE_K;
                // int B_idx = n_idx * m + m_idx;
                // int A_idx = m_idx * k + k_idx;
                // int A_idx = TILE_M * k_idx + 

                // Load quantized weights
                int A_q_idx = (m_iter * GLOBAL_M_SIZE * TILE_M * GLOBAL_K_SIZE +(GLOBAL_M_IDX * GLOBAL_K_SIZE + GLOBAL_K_IDX) * WEIGHT_TILE_SIZE + k_iter * GLOBAL_M_SIZE * TILE_M * GLOBAL_K_SIZE * TILE_K) / 4;
                #if WEIGHT_TILE_SIZE == 16
                    ushort4 A_q = vload4(0, src0_q + A_q_idx);
                #elif WEIGHT_TILE_SIZE == 32
                    ushort8 A_q = vload8(0, src0_q + A_q_idx);
                #elif WEIGHT_TILE_SIZE == 64
                    ushort16 A_q = vload16(0, src0_q + A_idx);
                #endif

                // Load scales
                int A_d_idx = m_idx + (k_idx / 32) * m;
                #if TILE_M == 2
                    half2 A_d = vload2(0, src0_d + A_d_idx);
                    half2 A;
                #elif TILE_M == 4
                    half4 A_d = vload4(0, src0_d + A_d_idx);
                    half4 A;
                #elif TILE_M == 8
                    half8 A_d = vload8(0, src0_d + A_d_idx);
                    half8 A;
                #elif TILE_M == 16
                    half16 A_d = vload16(0, src0_d + A_d_idx);
                    half16 A;
                #endif
                

                // Load inputs
                // half4 B = read_imageh(src1, gy*2 + (k_idx)*(n_4));
                int B_idx = (n_idx * m + m_idx) / 4;
                #if TILE_N == 1
                    DEFINE_B(0);
                    LOAD_B(0);
                #endif
                #if TILE_N >= 2
                    B_idx += n_idx * m / 4;
                    DEFINE_B(1);
                    LOAD_B(1);
                #endif
                #if TILE_N >= 4
                    B_idx += n_idx * m / 4;
                    DEFINE_B(2);
                    LOAD_B(2);
                    B_idx += n_idx * m / 4;
                    DEFINE_B(3);
                    LOAD_B(3);
                #endif
                #if TILE_N >= 8
                    B_idx += n_idx * m / 4;
                    DEFINE_B(4);
                    LOAD_B(4);
                    B_idx += n_idx * m / 4;
                    DEFINE_B(5);
                    LOAD_B(5);
                    B_idx += n_idx * m / 4;
                    DEFINE_B(6);
                    LOAD_B(6);
                    B_idx += n_idx * m / 4;
                    DEFINE_B(7);
                    LOAD_B(7);
                #endif

                // Compute
                COMPUTE;
            }

            // Reduce C

            // Store outputs
            #if GLOBAL_K_SIZE > 1
            if (WI_K_IDX == 0) {
            #endif
            #if TILE_N == 1
                #if TILE_M == 2
                    vstore2(convert_float2(C0), 0, dst + n_idx * m + m_idx);
                #elif TILE_M == 4
                    vstore4(convert_float4(C0), 0, dst + n_idx * m + m_idx);
                #elif TILE_M == 8
                    vstore8(convert_float8(C0), 0, dst + n_idx * m + m_idx);
                #elif TILE_M == 16
                    vstore16(convert_float16(C0), 0, dst + n_idx * m + m_idx);
                #endif
            #elif TILE_N == 2
                #if TILE_M == 2
                    vstore2(convert_float2(C0), 0, dst + n_idx * m + m_idx);
                    vstore2(convert_float2(C1), 0, dst + (n_idx + 1) * m + m_idx);
                #elif TILE_M == 4
                    vstore4(convert_float4(C0), 0, dst + n_idx * m + m_idx);
                    vstore4(convert_float4(C1), 0, dst + (n_idx + 1) * m + m_idx);
                #elif TILE_M == 8
                    vstore8(convert_float8(C0), 0, dst + n_idx * m + m_idx);
                    vstore8(convert_float8(C1), 0, dst + (n_idx + 1) * m + m_idx);
                #elif TILE_M == 16
                    vstore16(convert_float16(C0), 0, dst + n_idx * m + m_idx);
                    vstore16(convert_float16(C1), 0, dst + (n_idx + 1) * m + m_idx);
                #endif
            #elif TILE_N == 4
                #if TILE_M == 2
                    vstore2(convert_float2(C0), 0, dst + n_idx * m + m_idx);
                    vstore2(convert_float2(C1), 0, dst + (n_idx + 1) * m + m_idx);
                    vstore2(convert_float2(C2), 0, dst + (n_idx + 2) * m + m_idx);
                    vstore2(convert_float2(C3), 0, dst + (n_idx + 3) * m + m_idx);
                #elif TILE_M == 4
                    vstore4(convert_float4(C0), 0, dst + n_idx * m + m_idx);
                    vstore4(convert_float4(C1), 0, dst + (n_idx + 1) * m + m_idx);
                    vstore4(convert_float4(C2), 0, dst + (n_idx + 2) * m + m_idx);
                    vstore4(convert_float4(C3), 0, dst + (n_idx + 3) * m + m_idx);
                #elif TILE_M == 8
                    vstore8(convert_float8(C0), 0, dst + n_idx * m + m_idx);
                    vstore8(convert_float8(C1), 0, dst + (n_idx + 1) * m + m_idx);
                    vstore8(convert_float8(C2), 0, dst + (n_idx + 2) * m + m_idx);
                    vstore8(convert_float8(C3), 0, dst + (n_idx + 3) * m + m_idx);
                #elif TILE_M == 16
                    vstore16(convert_float16(C0), 0, dst + n_idx * m + m_idx);
                    vstore16(convert_float16(C1), 0, dst + (n_idx + 1) * m + m_idx);
                    vstore16(convert_float16(C2), 0, dst + (n_idx + 2) * m + m_idx);
                    vstore16(convert_float16(C3), 0, dst + (n_idx + 3) * m + m_idx);
                #endif
            #elif TILE_N == 8
                #if TILE_M == 2
                    vstore2(convert_float2(C0), 0, dst + n_idx * m + m_idx);
                    vstore2(convert_float2(C1), 0, dst + (n_idx + 1) * m + m_idx);
                    vstore2(convert_float2(C2), 0, dst + (n_idx + 2) * m + m_idx);
                    vstore2(convert_float2(C3), 0, dst + (n_idx + 3) * m + m_idx);
                    vstore2(convert_float2(C4), 0, dst + (n_idx + 4) * m + m_idx);
                    vstore2(convert_float2(C5), 0, dst + (n_idx + 5) * m + m_idx);
                    vstore2(convert_float2(C6), 0, dst + (n_idx + 6) * m + m_idx);
                    vstore2(convert_float2(C7), 0, dst + (n_idx + 7) * m + m_idx);
                #elif TILE_M == 4
                    vstore4(convert_float4(C0), 0, dst + n_idx * m + m_idx);
                    vstore4(convert_float4(C1), 0, dst + (n_idx + 1) * m + m_idx);
                    vstore4(convert_float4(C2), 0, dst + (n_idx + 2) * m + m_idx);
                    vstore4(convert_float4(C3), 0, dst + (n_idx + 3) * m + m_idx);
                    vstore4(convert_float4(C4), 0, dst + (n_idx + 4) * m + m_idx);
                    vstore4(convert_float4(C5), 0, dst + (n_idx + 5) * m + m_idx);
                    vstore4(convert_float4(C6), 0, dst + (n_idx + 6) * m + m_idx);
                    vstore4(convert_float4(C7), 0, dst + (n_idx + 7) * m + m_idx);
                #elif TILE_M == 8
                    vstore8(convert_float8(C0), 0, dst + n_idx * m + m_idx);
                    vstore8(convert_float8(C1), 0, dst + (n_idx + 1) * m + m_idx);
                    vstore8(convert_float8(C2), 0, dst + (n_idx + 2) * m + m_idx);
                    vstore8(convert_float8(C3), 0, dst + (n_idx + 3) * m + m_idx);
                    vstore8(convert_float8(C4), 0, dst + (n_idx + 4) * m + m_idx);
                    vstore8(convert_float8(C5), 0, dst + (n_idx + 5) * m + m_idx);
                    vstore8(convert_float8(C6), 0, dst + (n_idx + 6) * m + m_idx);
                    vstore8(convert_float8(C7), 0, dst + (n_idx + 7) * m + m_idx);
                #elif TILE_M == 16
                    vstore16(convert_float16(C0), 0, dst + n_idx * m + m_idx);
                    vstore16(convert_float16(C1), 0, dst + (n_idx + 1) * m + m_idx);
                    vstore16(convert_float16(C2), 0, dst + (n_idx + 2) * m + m_idx);
                    vstore16(convert_float16(C3), 0, dst + (n_idx + 3) * m + m_idx);
                    vstore16(convert_float16(C4), 0, dst + (n_idx + 4) * m + m_idx);
                    vstore16(convert_float16(C5), 0, dst + (n_idx + 5) * m + m_idx);
                    vstore16(convert_float16(C6), 0, dst + (n_idx + 6) * m + m_idx);
                    vstore16(convert_float16(C7), 0, dst + (n_idx + 7) * m + m_idx);
                #endif
            #endif
            #if GLOBAL_K_SIZE > 1
            }
            #endif
        }
    }
}

)";
}

#endif // GENERAL_GEMM_H
