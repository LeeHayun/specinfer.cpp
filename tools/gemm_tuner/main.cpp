#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <memory>
#include <cstring>
#include "general_gemm.h"
#include "half.hpp"


// OpenCL 에러 체크 매크로
#define CHECK_CL_ERROR(err) \
    if (err != CL_SUCCESS) { \
        std::cerr << "OpenCL 에러 " << err << " at line " << __LINE__ << std::endl; \
        return err; \
    }

// OpenCL 리소스 정리를 위한 RAII 클래스
class OpenCLContext {
private:
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    cl_program program_general_gemm;
    cl_kernel kernel_general_gemm;

    // data
    std::vector<float> input_data;
    std::vector<uint8_t> weight_data;
    std::vector<float> output_data;
    std::vector<float> scale_data;
    
    // 버퍼들
    cl_mem input_buffer;
    cl_mem input_image;
    cl_mem output_buffer;
    cl_mem weight_buffer;
    cl_mem scale_buffer;
    
    // 매트릭스 크기
    int M, N, K, N_padding, N_no_padding;
    size_t input_size, output_size, weight_size, scale_size;
    
    // 커널 이름
    std::string kernel_name;
    
public:
    OpenCLContext() : platform(0), device(0), context(0), queue(0), 
                      program_general_gemm(0), kernel_general_gemm(0),
                      input_buffer(0), output_buffer(0) {}
    
    ~OpenCLContext() {
        cleanup();
    }

    static cl_program build_program_from_source(cl_context ctx, cl_device_id dev, const char* program_buffer, const std::string &compile_opts) {
        cl_program p;
        char *program_log;
        size_t program_size;
        size_t log_size;
        int err;

        program_size = strlen(program_buffer);

        p = clCreateProgramWithSource(ctx, 1, (const char**)&program_buffer, &program_size, &err);
        if(err < 0) {
            std::cerr << "OpenCL error creating program" << std::endl;
            exit(1);
        }

        err = clBuildProgram(p, 0, NULL, compile_opts.c_str(), NULL, NULL);
        if(err < 0) {
            clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            program_log = (char*) malloc(log_size + 1);
            program_log[log_size] = '\0';
            clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
            std::cerr << "OpenCL error creating program:\n\n" << program_log << std::endl;
            free(program_log);
            exit(1);
        }

        return p;
    }
    
    int initialize(int m, int n, int k) {
        M = m; N = n; K = k;
        
        // OpenCL 플랫폼과 디바이스 초기화
        cl_int err;
        
        // 플랫폼 찾기
        err = clGetPlatformIDs(1, &platform, nullptr);
        CHECK_CL_ERROR(err);
        
        // GPU 디바이스 찾기
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        CHECK_CL_ERROR(err);
        
        // 컨텍스트 생성
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        CHECK_CL_ERROR(err);
        
        // 커맨드 큐 생성
        queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
        CHECK_CL_ERROR(err);
        
        // 프로그램 생성
        // std::string compile_opts = "-cl-mad-enagle -cl-unsafe-math-optimizations"
        //                            "-cl-finite-math-only -cl-fast-relaxed-math";
        // program_gemm_llama_cpp = build_program_from_source(context, device, OpenCLKernels::MUL_MAT_AB_BI_8X4_KERNEL_SOURCE.c_str(), compile_opts);        
        // program_gemm_mnn = build_program_from_source(context, device, OpenCLKernels::GEMM_KERNEL_SOURCE.c_str(), compile_opts);
        
        // GEMM 커널 생성
        // kernel_gemm_llama_cpp = clCreateKernel(program_gemm_llama_cpp, "kernel_mul_mat_Ab_Bi_8x4", &err);
        // CHECK_CL_ERROR(err);
        // kernel_gemm_mnn = clCreateKernel(program_gemm_mnn, "gemm_b4_c8_int4_buf", &err);
        // CHECK_CL_ERROR(err);
        
        // 버퍼 크기 계산 (gemm_c4nhw4_to_nhwc용)
        input_size = M * N * sizeof(float);
        output_size = M * N * sizeof(float);
        weight_size = M * K / 2;
        scale_size = M * K / 32 * 2;
        
        // 버퍼 생성 (gemm_c4nhw4_to_nhwc용)
        input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, input_size, nullptr, &err);
        CHECK_CL_ERROR(err);
        
        output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size, nullptr, &err);
        CHECK_CL_ERROR(err);

        weight_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, weight_size, nullptr, &err);
        CHECK_CL_ERROR(err);

        scale_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, scale_size, nullptr, &err);
        CHECK_CL_ERROR(err);
        

        cl_image_format img_fmt_1d;
        cl_image_desc img_desc_1d;
        img_fmt_1d = {CL_RGBA, CL_FLOAT};
        memset(&img_desc_1d, 0, sizeof(img_desc_1d));
        img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc_1d.image_width = N * K / 4;
        img_desc_1d.buffer = input_buffer;        
        input_image = clCreateImage(
            context,
            CL_MEM_READ_ONLY,
            &img_fmt_1d,
            &img_desc_1d,
            NULL,
            &err);
        CHECK_CL_ERROR(err);
        
        // 테스트 데이터 초기화
        initializeTestData();
        
        return 0;
    }
    
    void cleanup() {
        if (output_buffer) clReleaseMemObject(output_buffer);
        if (input_buffer) clReleaseMemObject(input_buffer);
        if (program_general_gemm) clReleaseProgram(program_general_gemm);
        if (kernel_general_gemm) clReleaseKernel(kernel_general_gemm);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }
    
    double runKernel(cl_kernel kernel, size_t global_size[3], size_t local_size[3]) {
        cl_int err;
        
        // 커널 인자 설정 (gemm_c4nhw4_to_nhwc용)
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &weight_buffer);
        CHECK_CL_ERROR(err);
        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &scale_buffer);
        CHECK_CL_ERROR(err);
        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &input_image);
        CHECK_CL_ERROR(err);
        err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &output_buffer);
        CHECK_CL_ERROR(err);
        err = clSetKernelArg(kernel, 4, sizeof(int), &M);
        CHECK_CL_ERROR(err);
        err = clSetKernelArg(kernel, 5, sizeof(int), &N_padding);
        CHECK_CL_ERROR(err);
        err = clSetKernelArg(kernel, 6, sizeof(int), &K);
        CHECK_CL_ERROR(err);
        err = clSetKernelArg(kernel, 7, sizeof(int), &N_no_padding);
        CHECK_CL_ERROR(err);
        
        // 이벤트 생성
        cl_event event;
        
        // 커널 실행
        err = clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, global_size, local_size, 0, nullptr, &event);
        CHECK_CL_ERROR(err);
        
        // 완료 대기
        err = clWaitForEvents(1, &event);
        CHECK_CL_ERROR(err);
        
        // 실행 시간 측정
        cl_ulong start_time, end_time;
        err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, nullptr);
        CHECK_CL_ERROR(err);
        err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, nullptr);
        CHECK_CL_ERROR(err);
        
        clReleaseEvent(event);
        
        return (end_time - start_time) / 1000000.0; // ms 단위로 변환
    }
    
    void tuneKernel() {
        std::cout << "=== OpenCL GEMM Work Size 튜닝 ===" << std::endl;
        std::cout << "매트릭스 크기: " << M << " x " << N << " x " << K << std::endl;
        
        // 디바이스 정보 출력
        size_t max_work_group_size;
        cl_int err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, 
                                    sizeof(size_t), &max_work_group_size, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "디바이스 정보 조회 실패" << std::endl;
            return;
        }
        
        size_t max_work_item_sizes[3];
        err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, 
                             sizeof(max_work_item_sizes), max_work_item_sizes, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "디바이스 정보 조회 실패" << std::endl;
            return;
        }

        double best_time = std::numeric_limits<double>::max();
        double best_gflops = 0.0;
        size_t count = 0;

        for (size_t TILE_M = 2; TILE_M <= std::max(M, 16); TILE_M *= 2) {
            for (size_t TILE_K = 1; TILE_K <= std::max(K, 32); TILE_K *= 2) {
                size_t weight_tile_size = TILE_M * TILE_K;
                if (weight_tile_size < 16 || weight_tile_size > 64) {
                    continue;
                }
                for (size_t TILE_N = 1; TILE_N <= std::max(N, 8); TILE_N *= 2) {
                    for (size_t WG_N = 1; WG_N <= N / TILE_N; WG_N *= 2) {
                        for (size_t WG_M = 1; WG_M <= M / TILE_M; WG_M *= 2) {
                            for (size_t WI_N = 1; WI_N <= N / (WG_N * TILE_N); WI_N *= 2) {
                                for (size_t WI_M = 1; WI_M <= M / (WG_M * TILE_M); WI_M *= 2) {
                                    for (size_t WI_K = 1; WI_K <= K / TILE_K; WI_K *= 2) {
                                        size_t num_thread_per_group = WI_N * WI_M * WI_K;
                                        size_t total_thread = num_thread_per_group * WG_N * WG_M;
                                        if (total_thread < 1024 || num_thread_per_group < 64 || num_thread_per_group > 1024) {
                                            continue;
                                        }
                                        count += 1;
                                        continue;
                                        size_t N_ITER = N / (WG_N * WI_N * TILE_N);
                                        size_t M_ITER = M / (WG_M * WI_M * TILE_M);
                                        size_t K_ITER = K / (WI_K * TILE_K);
                                        std::cout << "TILE_N: " << TILE_N << ", TILE_M: " << TILE_M << ", TILE_K: " << TILE_K << ", WG_N: " << WG_N << ", WG_M: " << WG_M << ", WI_N: " << WI_N << ", WI_M: " << WI_M << ", WI_K: " << WI_K << std::endl;
                                        std::cout << "N_ITER: " << N_ITER << ", M_ITER: " << M_ITER << ", K_ITER: " << K_ITER << std::endl;

                                        std::string compile_opts = "-cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math ";
                                        compile_opts += " -D TILE_N=" + std::to_string(TILE_N);
                                        compile_opts += " -D TILE_M=" + std::to_string(TILE_M);
                                        compile_opts += " -D TILE_K=" + std::to_string(TILE_K);
                                        compile_opts += " -D WG_N=" + std::to_string(WG_N);
                                        compile_opts += " -D WG_M=" + std::to_string(WG_M);
                                        compile_opts += " -D WI_N=" + std::to_string(WI_N);
                                        compile_opts += " -D WI_M=" + std::to_string(WI_M);
                                        compile_opts += " -D DIM_N=1";
                                        compile_opts += " -D DIM_M=2";
                                        compile_opts += " -D DIM_K=0";
                                        compile_opts += " -D M=" + std::to_string(M);
                                        compile_opts += " -D N=" + std::to_string(N);
                                        compile_opts += " -D K=" + std::to_string(K);
                                        compile_opts += " -D WEIGHT_TILE_SIZE=" + std::to_string(TILE_M * TILE_K);
                                        compile_opts += " -D TILE_M_x_GLOBAL_M_SIZE=" + std::to_string(TILE_M * WG_M * WI_M);
                                        compile_opts += " -D TILE_N_x_GLOBAL_N_SIZE=" + std::to_string(TILE_N * WG_N * WI_N);
                                        compile_opts += " -D GLOBAL_N_SIZE=" + std::to_string(WG_N * WI_N);
                                        compile_opts += " -D GLOBAL_M_SIZE=" + std::to_string(WG_M * WI_M);
                                        compile_opts += " -D GLOBAL_K_SIZE=" + std::to_string(WI_K);

                                        program_general_gemm = build_program_from_source(context, device, OpenCLKernels::GENERAL_GEMM_KERNEL_SOURCE.c_str(), compile_opts);
                                        kernel_general_gemm = clCreateKernel(program_general_gemm, "kernel_mul_mat_q4_0_f32", &err);
                                        if (err != CL_SUCCESS) {
                                            std::cerr << "커널 생성 실패" << std::endl;
                                            return;
                                        }

                                        size_t global_size[3] = {WI_K, WG_N*WI_N, WG_M*WI_M};
                                        size_t local_size[3] = {WI_K, WI_N, WI_M};
                                        N_padding = (N + TILE_M - 1) / TILE_M * TILE_M;
                                        N_no_padding = N;

                                        double time = runKernel(kernel_general_gemm, global_size, local_size);
                                        double gflops = (2.0 * M * N * K) / (time * 1e6);
                                        if (gflops > best_gflops) {
                                            best_gflops = gflops;
                                            best_time = time;
                                        }
                                        std::cout << "Time: " << time << " ms, GFLOPS: " << gflops << std::endl;
                                        std::cout << "Best Time: " << best_time << " ms, Best GFLOPS: " << best_gflops << std::endl;

                                        clReleaseKernel(kernel_general_gemm);
                                        clReleaseProgram(program_general_gemm);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }   
        std::cout << "Total count: " << count << std::endl;
        
        // std::cout << "최대 워크 그룹 크기: " << max_work_group_size << std::endl;
        // std::cout << "최대 워크 아이템 크기: [" << max_work_item_sizes[0] << ", " 
        //           << max_work_item_sizes[1] << ", " << max_work_item_sizes[2] << "]" << std::endl;
        
        // // 튜닝할 work size 범위 정의
        // std::vector<size_t> local_sizes_x = {8, 16, 32, 64, 128};
        // std::vector<size_t> local_sizes_y = {8, 16, 32, 64, 128};
        
        // // Global work size 계산 (커널에 맞게 조정)
        // size_t global_size_x = ((M + 3) / 4) * 4; // b/4 * 4
        // size_t global_size_y = ((N + 7) / 8) * 8; // c/8 * 8
        
        // std::cout << "Global work size: [" << global_size_x << ", " << global_size_y << "]" << std::endl;
        
        // double best_time = std::numeric_limits<double>::max();
        // size_t best_local_x = 0, best_local_y = 0;
        
        // std::vector<std::tuple<size_t, size_t, double>> results;
        
        // // 모든 조합 테스트
        // for (size_t local_x : local_sizes_x) {
        //     for (size_t local_y : local_sizes_y) {
        //         // 워크 그룹 크기 제한 확인
        //         if (local_x * local_y > max_work_group_size) {
        //             continue;
        //         }
                
        //         // 워크 아이템 크기 제한 확인
        //         if (local_x > max_work_item_sizes[0] || local_y > max_work_item_sizes[1]) {
        //             continue;
        //         }
                
        //         // Global size가 local size로 나누어 떨어지는지 확인
        //         if (global_size_x % local_x != 0 || global_size_y % local_y != 0) {
        //             continue;
        //         }
                
        //         std::cout << "테스트 중: local_size = [" << local_x << ", " << local_y << "] ";
                
        //         // 여러 번 실행하여 평균 시간 측정
        //         const int num_runs = 10;
        //         double total_time = 0.0;
                
        //         for (int i = 0; i < num_runs; i++) {
        //             double time = runKernel(global_size_x, global_size_y, local_x, local_y);
        //             total_time += time;
        //         }
                
        //         double avg_time = total_time / num_runs;
        //         results.push_back({local_x, local_y, avg_time});
                
        //         std::cout << "평균 시간: " << std::fixed << std::setprecision(3) << avg_time << " ms" << std::endl;
                
        //         if (avg_time < best_time) {
        //             best_time = avg_time;
        //             best_local_x = local_x;
        //             best_local_y = local_y;
        //         }
        //     }
        // }
        
        // // 결과 정렬 및 출력
        // std::sort(results.begin(), results.end(), 
        //           [](const std::tuple<size_t, size_t, double>& a, 
        //              const std::tuple<size_t, size_t, double>& b) { 
        //               return std::get<2>(a) < std::get<2>(b); 
        //           });
        
        // std::cout << "\n=== 튜닝 결과 ===" << std::endl;
        // std::cout << std::setw(10) << "Local X" << std::setw(10) << "Local Y" 
        //           << std::setw(15) << "시간 (ms)" << std::setw(15) << "성능 (GFLOPS)" << std::endl;
        // std::cout << std::string(50, '-') << std::endl;
        
        // for (const auto& result : results) {
        //     size_t local_x = std::get<0>(result);
        //     size_t local_y = std::get<1>(result);
        //     double time = std::get<2>(result);
            
        //     // GFLOPS 계산 (GEMM: 2*M*N*K operations)
        //     double gflops = (2.0 * M * N * K) / (time * 1e6);
            
        //     std::cout << std::setw(10) << local_x 
        //               << std::setw(10) << local_y 
        //               << std::setw(15) << std::fixed << std::setprecision(3) << time
        //               << std::setw(15) << std::fixed << std::setprecision(2) << gflops << std::endl;
        // }
        
        // std::cout << "\n최적 설정:" << std::endl;
        // std::cout << "  Local work size: [" << best_local_x << ", " << best_local_y << "]" << std::endl;
        // std::cout << "  Global work size: [" << global_size_x << ", " << global_size_y << "]" << std::endl;
        // std::cout << "  최적 시간: " << std::fixed << std::setprecision(3) << best_time << " ms" << std::endl;
        // double best_gflops = (2.0 * M * N * K) / (best_time * 1e6);
        // std::cout << "  최적 성능: " << std::fixed << std::setprecision(2) << best_gflops << " GFLOPS" << std::endl;
    }
    
private:
    void initializeTestData() {
        input_data.resize(N * K);
        weight_data.resize(M * K / 2);
        output_data.resize(M * N);
        scale_data.resize(M * N / 4);

        // 입력 데이터 초기화 (gemm_c4nhw4_to_nhwc용)
        std::vector<float> input_data(M * N);
        for (int i = 0; i < M * N; i++) {
            input_data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
        
        // 버퍼에 데이터 쓰기
        cl_int err;
        err = clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, input_size, input_data.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "입력 버퍼 쓰기 실패" << std::endl;
            return;
        }
    }
};

int main(int argc, char* argv[]) {
    // 매트릭스 크기 설정 (기본값)
    int M = 1024;  // 배치 크기
    int N = 1024;  // 출력 채널
    int K = 1024;  // 입력 채널
    
    // 명령행 인수 처리
    if (argc >= 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    } else if (argc > 1) {
        std::cout << "사용법: " << argv[0] << " [M] [N] [K]" << std::endl;
        std::cout << "  M: 출력 채널 (기본값: 1024)" << std::endl;
        std::cout << "  N: 배치 크기 (기본값: 1024)" << std::endl;
        std::cout << "  K: 입력 채널 (기본값: 1024)" << std::endl;
        std::cout << "예시: " << argv[0] << " 512 1024 2048" << std::endl;
        return 0;
    }
    
    std::cout << "매트릭스 크기: " << M << " x " << N << " x " << K << std::endl;
    
    OpenCLContext context;
    
    if (context.initialize(M, N, K) != 0) {
        std::cerr << "OpenCL 초기화 실패" << std::endl;
        return -1;
    }
    
    // Work size 튜닝 실행
    context.tuneKernel();
    
    return 0;
}
