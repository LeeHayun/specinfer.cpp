#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// 16-bit transpose, loading/storing a 4x4 tile of elements
kernel void kernel_transpose_16(
    __read_only image1d_buffer_t input,
    __write_only image1d_buffer_t output,
    const uint rows,
    const uint cols
) {

    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int i_2 = i<<2;
    const int j_2 = j<<2;

    half4 temp0 = read_imageh(input, (j_2+0)*cols+i);
    half4 temp1 = read_imageh(input, (j_2+1)*cols+i);
    half4 temp2 = read_imageh(input, (j_2+2)*cols+i);
    half4 temp3 = read_imageh(input, (j_2+3)*cols+i);

    write_imageh(output, (i_2+0)*rows+j, (half4)(temp0.s0, temp1.s0, temp2.s0, temp3.s0));
    write_imageh(output, (i_2+1)*rows+j, (half4)(temp0.s1, temp1.s1, temp2.s1, temp3.s1));
    write_imageh(output, (i_2+2)*rows+j, (half4)(temp0.s2, temp1.s2, temp2.s2, temp3.s2));
    write_imageh(output, (i_2+3)*rows+j, (half4)(temp0.s3, temp1.s3, temp2.s3, temp3.s3));
}

// 32-bit transpose, loading/storing a 4x4 tile of elements
kernel void kernel_transpose_32(
    __read_only image1d_buffer_t input,
    __write_only image1d_buffer_t output,
    const uint rows,
    const uint cols
) {

    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int i_2 = i<<2;
    const int j_2 = j<<2;

    float4 temp0 = read_imagef(input, (j_2+0)*cols+i);
    float4 temp1 = read_imagef(input, (j_2+1)*cols+i);
    float4 temp2 = read_imagef(input, (j_2+2)*cols+i);
    float4 temp3 = read_imagef(input, (j_2+3)*cols+i);

    write_imagef(output, (i_2+0)*rows+j, (float4)(temp0.s0, temp1.s0, temp2.s0, temp3.s0));
    write_imagef(output, (i_2+1)*rows+j, (float4)(temp0.s1, temp1.s1, temp2.s1, temp3.s1));
    write_imagef(output, (i_2+2)*rows+j, (float4)(temp0.s2, temp1.s2, temp2.s2, temp3.s2));
    write_imagef(output, (i_2+3)*rows+j, (float4)(temp0.s3, temp1.s3, temp2.s3, temp3.s3));

}

// 32-bit transpose, loading/storing a 4x4 tile of elements
// Only used for activations
// converts to FP16
// also adds zero padding for non multiple of 8 prompt lengths
kernel void kernel_transpose_32_16(__read_only image1d_buffer_t input, __write_only image1d_buffer_t output, const uint rows, const uint cols, const uint padded_rows) {

    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int i_2 = i<<2;
    const int j_2 = j<<2;
    half4 temp0 = {0,0,0,0}; // initialize outputs to 0
    half4 temp1 = {0,0,0,0};
    half4 temp2 = {0,0,0,0};
    half4 temp3 = {0,0,0,0};

    if((j_2+0)*cols+i*4+3 < rows*cols*16){ // only load from a valid location. Otherwise keep register data as 0
        temp0 = read_imageh(input, (j_2+0)*cols+i);
    }
    if((j_2+1)*cols+i*4+3 < rows*cols*16){
        temp1 = read_imageh(input, (j_2+1)*cols+i);
    }
    if((j_2+2)*cols+i*4+3 < rows*cols*16){
        temp2 = read_imageh(input, (j_2+2)*cols+i);
    }
    if((j_2+3)*cols+i*4+3 < rows*cols*16){
        temp3 = read_imageh(input, (j_2+3)*cols+i);
    }

    write_imageh(output, (i_2+0)*padded_rows+j, (half4)(temp0.s0, temp1.s0, temp2.s0, temp3.s0)); // no conditionals for output, includes zero padding
    write_imageh(output, (i_2+1)*padded_rows+j, (half4)(temp0.s1, temp1.s1, temp2.s1, temp3.s1));
    write_imageh(output, (i_2+2)*padded_rows+j, (half4)(temp0.s2, temp1.s2, temp2.s2, temp3.s2));
    write_imageh(output, (i_2+3)*padded_rows+j, (half4)(temp0.s3, temp1.s3, temp2.s3, temp3.s3));
}

// loading/storing a 4x8 tile of elements
kernel void kernel_transpose_q4_0(
    __read_only image1d_buffer_t input,
    __write_only image1d_buffer_t output,
    const uint rows,
    const uint cols
) {
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int i_2 = i<<2;
    const int j_3 = j<<3;

    half4 temp0 = read_imageh(input, (j_3+0)*cols+i);
    half4 temp1 = read_imageh(input, (j_3+1)*cols+i);
    half4 temp2 = read_imageh(input, (j_3+2)*cols+i);
    half4 temp3 = read_imageh(input, (j_3+3)*cols+i);
    half4 temp4 = read_imageh(input, (j_3+4)*cols+i);
    half4 temp5 = read_imageh(input, (j_3+5)*cols+i);
    half4 temp6 = read_imageh(input, (j_3+6)*cols+i);
    half4 temp7 = read_imageh(input, (j_3+7)*cols+i);

    ushort4 temp0_ushort = as_ushort4(temp0);
    ushort4 temp1_ushort = as_ushort4(temp1);
    ushort4 temp2_ushort = as_ushort4(temp2);
    ushort4 temp3_ushort = as_ushort4(temp3);
    ushort4 temp4_ushort = as_ushort4(temp4);
    ushort4 temp5_ushort = as_ushort4(temp5);
    ushort4 temp6_ushort = as_ushort4(temp6);
    ushort4 temp7_ushort = as_ushort4(temp7);

    ushort4 result0_ushort = (ushort4)(
        (temp0_ushort.s0 & 0x000F) | (temp1_ushort.s0 & 0x000F) << 4 | (temp2_ushort.s0 & 0x000F) << 8 | (temp3_ushort.s0 & 0x000F) << 12,
        (temp4_ushort.s0 & 0x000F) | (temp5_ushort.s0 & 0x000F) << 4 | (temp6_ushort.s0 & 0x000F) << 8 | (temp7_ushort.s0 & 0x000F) << 12,
        (temp0_ushort.s0 & 0x00F0) >> 4 | (temp1_ushort.s0 & 0x00F0) | (temp2_ushort.s1 & 0x00F0) << 4 | (temp3_ushort.s1 & 0x00F0) << 8,
        (temp4_ushort.s0 & 0x00F0) >> 4 | (temp5_ushort.s0 & 0x00F0) | (temp6_ushort.s1 & 0x00F0) << 4 | (temp7_ushort.s1 & 0x00F0) << 8
    );
    ushort4 result1_ushort = (ushort4)(
        (temp0_ushort.s0 & 0x0F00) >> 8 | (temp1_ushort.s0 & 0x0F00) >> 4 | (temp2_ushort.s0 & 0x0F00) | (temp3_ushort.s0 & 0x0F00) << 4,
        (temp4_ushort.s0 & 0x0F00) >> 8 | (temp5_ushort.s0 & 0x0F00) >> 4 | (temp6_ushort.s0 & 0x0F00) | (temp7_ushort.s0 & 0x0F00) << 4,
        (temp0_ushort.s0 & 0xF000) >> 12 | (temp1_ushort.s0 & 0xF000) >> 8 | (temp2_ushort.s0 & 0xF000) >> 4 | (temp3_ushort.s0 & 0xF000),
        (temp4_ushort.s0 & 0xF000) >> 12 | (temp5_ushort.s0 & 0xF000) >> 8 | (temp6_ushort.s0 & 0xF000) >> 4 | (temp7_ushort.s0 & 0xF000)
    );
    ushort4 result2_ushort = (ushort4)(
        (temp0_ushort.s1 & 0x000F) | (temp1_ushort.s1 & 0x000F) << 4 | (temp2_ushort.s1 & 0x000F) << 8 | (temp3_ushort.s1 & 0x000F) << 12,
        (temp4_ushort.s1 & 0x000F) | (temp5_ushort.s1 & 0x000F) << 4 | (temp6_ushort.s1 & 0x000F) << 8 | (temp7_ushort.s1 & 0x000F) << 12,
        (temp0_ushort.s1 & 0x00F0) >> 4 | (temp1_ushort.s1 & 0x00F0) | (temp2_ushort.s1 & 0x00F0) << 4 | (temp3_ushort.s1 & 0x00F0) << 8,
        (temp4_ushort.s1 & 0x00F0) >> 4 | (temp5_ushort.s1 & 0x00F0) | (temp6_ushort.s1 & 0x00F0) << 4 | (temp7_ushort.s1 & 0x00F0) << 8
    );
    ushort4 result3_ushort = (ushort4)(
        (temp0_ushort.s1 & 0x0F00) >> 8 | (temp1_ushort.s1 & 0x0F00) >> 4 | (temp2_ushort.s1 & 0x0F00) | (temp3_ushort.s1 & 0x0F00) << 4,
        (temp4_ushort.s1 & 0x0F00) >> 8 | (temp5_ushort.s1 & 0x0F00) >> 4 | (temp6_ushort.s1 & 0x0F00) | (temp7_ushort.s1 & 0x0F00) << 4,
        (temp0_ushort.s1 & 0xF000) >> 12 | (temp1_ushort.s1 & 0xF000) >> 8 | (temp2_ushort.s1 & 0xF000) >> 4 | (temp3_ushort.s1 & 0xF000),
        (temp4_ushort.s1 & 0xF000) >> 12 | (temp5_ushort.s1 & 0xF000) >> 8 | (temp6_ushort.s1 & 0xF000) >> 4 | (temp7_ushort.s1 & 0xF000)
    );
    ushort4 result4_ushort = (ushort4)(
        (temp0_ushort.s2 & 0x000F) | (temp1_ushort.s2 & 0x000F) << 4 | (temp2_ushort.s2 & 0x000F) << 8 | (temp3_ushort.s2 & 0x000F) << 12,
        (temp4_ushort.s2 & 0x000F) | (temp5_ushort.s2 & 0x000F) << 4 | (temp6_ushort.s2 & 0x000F) << 8 | (temp7_ushort.s2 & 0x000F) << 12,
        (temp0_ushort.s2 & 0x00F0) >> 4 | (temp1_ushort.s2 & 0x00F0) | (temp2_ushort.s2 & 0x00F0) << 4 | (temp3_ushort.s2 & 0x00F0) << 8,
        (temp4_ushort.s2 & 0x00F0) >> 4 | (temp5_ushort.s2 & 0x00F0) | (temp6_ushort.s2 & 0x00F0) << 4 | (temp7_ushort.s2 & 0x00F0) << 8
    );
    ushort4 result5_ushort = (ushort4)(
        (temp0_ushort.s2 & 0x0F00) >> 8 | (temp1_ushort.s2 & 0x0F00) >> 4 | (temp2_ushort.s2 & 0x0F00) | (temp3_ushort.s2 & 0x0F00) << 4,
        (temp4_ushort.s2 & 0x0F00) >> 8 | (temp5_ushort.s2 & 0x0F00) >> 4 | (temp6_ushort.s2 & 0x0F00) | (temp7_ushort.s2 & 0x0F00) << 4,
        (temp0_ushort.s2 & 0xF000) >> 12 | (temp1_ushort.s2 & 0xF000) >> 8 | (temp2_ushort.s2 & 0xF000) >> 4 | (temp3_ushort.s2 & 0xF000),
        (temp4_ushort.s2 & 0xF000) >> 12 | (temp5_ushort.s2 & 0xF000) >> 8 | (temp6_ushort.s2 & 0xF000) >> 4 | (temp7_ushort.s2 & 0xF000)
    );
    ushort4 result6_ushort = (ushort4)(
        (temp0_ushort.s3 & 0x000F) | (temp1_ushort.s3 & 0x000F) << 4 | (temp2_ushort.s3 & 0x000F) << 8 | (temp3_ushort.s3 & 0x000F) << 12,
        (temp4_ushort.s3 & 0x000F) | (temp5_ushort.s3 & 0x000F) << 4 | (temp6_ushort.s3 & 0x000F) << 8 | (temp7_ushort.s3 & 0x000F) << 12,
        (temp0_ushort.s3 & 0x00F0) >> 4 | (temp1_ushort.s3 & 0x00F0) | (temp2_ushort.s3 & 0x00F0) << 4 | (temp3_ushort.s3 & 0x00F0) << 8,
        (temp4_ushort.s3 & 0x00F0) >> 4 | (temp5_ushort.s3 & 0x00F0) | (temp6_ushort.s3 & 0x00F0) << 4 | (temp7_ushort.s3 & 0x00F0) << 8
    );
    ushort4 result7_ushort = (ushort4)(
        (temp0_ushort.s3 & 0x0F00) >> 8 | (temp1_ushort.s3 & 0x0F00) >> 4 | (temp2_ushort.s3 & 0x0F00) | (temp3_ushort.s3 & 0x0F00) << 4,
        (temp4_ushort.s3 & 0x0F00) >> 8 | (temp5_ushort.s3 & 0x0F00) >> 4 | (temp6_ushort.s3 & 0x0F00) | (temp7_ushort.s3 & 0x0F00) << 4,
        (temp0_ushort.s3 & 0xF000) >> 12 | (temp1_ushort.s3 & 0xF000) >> 8 | (temp2_ushort.s3 & 0xF000) >> 4 | (temp3_ushort.s3 & 0xF000),
        (temp4_ushort.s3 & 0xF000) >> 12 | (temp5_ushort.s3 & 0xF000) >> 8 | (temp6_ushort.s3 & 0xF000) >> 4 | (temp7_ushort.s3 & 0xF000)
    );

    half4 result0 = as_half4(result0_ushort);
    half4 result1 = as_half4(result1_ushort);
    half4 result2 = as_half4(result2_ushort);
    half4 result3 = as_half4(result3_ushort);
    half4 result4 = as_half4(result4_ushort);
    half4 result5 = as_half4(result5_ushort);
    half4 result6 = as_half4(result6_ushort);
    half4 result7 = as_half4(result7_ushort);


    write_imageh(output, (j*cols+i)*8+0, result0);
    write_imageh(output, (j*cols+i)*8+1, result1);
    write_imageh(output, (j*cols+i)*8+2, result2);
    write_imageh(output, (j*cols+i)*8+3, result3);
    write_imageh(output, (j*cols+i)*8+4, result4);
    write_imageh(output, (j*cols+i)*8+5, result5);
    write_imageh(output, (j*cols+i)*8+6, result6);
    write_imageh(output, (j*cols+i)*8+7, result7);

}