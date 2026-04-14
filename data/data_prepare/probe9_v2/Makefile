# Makefile for tile_probe

CXX = g++
CXXFLAGS = -O3 -fopenmp -Wall
LDFLAGS = -fopenmp -lm

TARGET = tile_probe
SRCS = tile_probe.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(SRCS) mmio.h
	$(CXX) $(CXXFLAGS) -o $@ $(SRCS) $(LDFLAGS)

clean:
	rm -f $(TARGET) $(OBJS)

# 测试运行 (需要指定矩阵文件)
test:
	@echo "Usage: ./tile_probe <matrix.mtx>"
	@echo "Example: ./tile_probe ../mtx/mc2depi.mtx"

.PHONY: all clean test
