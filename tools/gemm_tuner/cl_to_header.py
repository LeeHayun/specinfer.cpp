#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

def cl_to_header(cl_file, header_file):
    """
    OpenCL .cl 파일을 C++ 헤더 파일로 변환합니다.
    """
    try:
        # .cl 파일 읽기
        with open(cl_file, 'r', encoding='utf-8') as f:
            cl_content = f.read()

        # kernel name
        kernel_name = cl_file.replace('.cl', '').upper()
        
        # 헤더 가드 생성
        header_guard = os.path.basename(header_file).upper().replace('.', '_')
        
        # C++ 헤더 파일 생성
        header_content = f"""// Auto-generated from {cl_file}
// DO NOT EDIT - 이 파일은 자동 생성되었습니다.

#ifndef {header_guard}
#define {header_guard}

#include <string>

namespace OpenCLKernels {{
    const std::string {kernel_name}_KERNEL_SOURCE = R"(
{cl_content}
)";
}}

#endif // {header_guard}
"""
        
        # 헤더 파일 쓰기
        with open(header_file, 'w', encoding='utf-8') as f:
            f.write(header_content)
        
        print(f"✅ {cl_file} -> {header_file} 변환 완료")
        return True
        
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {cl_file}")
        return False
    except Exception as e:
        print(f"❌ 변환 중 오류 발생: {e}")
        return False

def main():
    if len(sys.argv) == 3:
        # 단일 파일 변환
        cl_file = sys.argv[1]
        header_file = sys.argv[2]
        
        if cl_to_header(cl_file, header_file):
            sys.exit(0)
        else:
            sys.exit(1)
    elif len(sys.argv) == 1:
        # 모든 .cl 파일을 헤더로 변환
        cl_files = ["gemm.cl", "mul_mat_Ab_Bi_8x4.cl"]
        
        for cl_file in cl_files:
            if os.path.exists(cl_file):
                header_file = cl_file.replace('.cl', '_kernel.h')
                if cl_to_header(cl_file, header_file):
                    print(f"✅ {cl_file} -> {header_file} 변환 완료")
                else:
                    print(f"❌ {cl_file} 변환 실패")
            else:
                print(f"⚠️ 파일을 찾을 수 없습니다: {cl_file}")
        
        sys.exit(0)
    else:
        print("사용법:")
        print("  python cl_to_header.py <input.cl> <output.h>  # 단일 파일 변환")
        print("  python cl_to_header.py                        # 모든 .cl 파일 변환")
        sys.exit(1)

if __name__ == "__main__":
    main() 