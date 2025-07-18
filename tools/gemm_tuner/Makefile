# === 설정 ===
NDK_PATH ?= $(ANDROID_NDK)
TOOLCHAIN := $(NDK_PATH)/toolchains/llvm/prebuilt/linux-x86_64

ifeq ($(NDK_PATH),)
	$(error NDK_PATH is not set)
endif

TARGET := aarch64-linux-android
API_LEVEL := 24
CXX := $(TOOLCHAIN)/bin/$(TARGET)$(API_LEVEL)-clang++
SYSROOT := $(TOOLCHAIN)/sysroot

# === 파일 설정 ===
SRC := main.cpp
OUT := main
OBJ_DIR := obj
BIN_DIR := bin

CFLAGS := -O2 --target=$(TARGET) --sysroot=$(SYSROOT) -fPIE -fPIC
LDFLAGS := -pie -llog -lOpenCL

# === 타겟 ===
all: $(BIN_DIR)/$(OUT)

$(BIN_DIR)/$(OUT): $(SRC)
	mkdir -p $(BIN_DIR)
	$(CXX) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean
