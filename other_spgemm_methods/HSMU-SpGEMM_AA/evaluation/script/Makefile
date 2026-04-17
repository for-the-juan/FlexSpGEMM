CXX := nvcc
#Replace cuda path with actual cuda path
CXXFLAGS := -arch=compute_80 -code=sm_80 -O3 -Xcompiler -lrt -lrt -lcudart -lcusparse -I/usr/local/cuda-11.8/include/cub

#Replace with the actual file path
SRC := ../../src/test.cu
TARGET := test

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@
clean:
	rm -f $(TARGET)
