# Compiler and flags
CXX = mpicxx
CXXFLAGS = -O3 -Wall -std=c++17
LDFLAGS = -libverbs

# Target executable
TARGET = dht_count

# Source file
SRC = dht_count.cpp

# Default target
all: $(TARGET)

# Build rule
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Run rule - accepts input and query files as arguments
run: $(TARGET)
	@if [ "$(input)" = "" ] || [ "$(query)" = "" ]; then \
		echo "Usage: make run input=<input_file> query=<query_file>"; \
		echo "Example: make run input=test1.txt query=query.txt"; \
		exit 1; \
	fi
	@mpirun -n 8 --hostfile hostfile --mca btl_tcp_if_include ibp216s0f0 ./$(TARGET) $(input) $(query)

# Clean rule
clean:
	rm -f $(TARGET)

# Help target
help:
	@echo "Available commands:"
	@echo "  make              - Build the program"
	@echo "  make clean        - Remove executable"
	@echo "  make run input=<input_file> query=<query_file>"
	@echo "                    - Run the program with specified input and query files"
	@echo "                    - Example: make run input=test1.txt query=query.txt"

# Allow arguments to be passed to the run target
%:
	@:

.PHONY: all clean run help