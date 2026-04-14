CXX = g++
CXXFLAGS = -O3 -fopenmp -Wall
LDFLAGS = -fopenmp -lm

TARGET = probeC
SRCS = probeC.cpp

all: $(TARGET)

$(TARGET): $(SRCS) mmio.h
	$(CXX) $(CXXFLAGS) -o $@ $(SRCS) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
