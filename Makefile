# Compiler
CXX = g++
# Compiler flags (includes OpenMP)
CXXFLAGS = -std=c++11 -Wall -O2 -fopenmp
# Target executable
TARGET = correlate
# Source files
SOURCES = main.cpp correlate.cpp
# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Default target
all: $(TARGET)

# Link object files
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

# Compile .cpp to .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean
clean:
	rm -f $(OBJECTS) $(TARGET)

# Run (usage: make run NY=200 NX=500 THREADS=4)
run: $(TARGET)
	OMP_NUM_THREADS=$(THREADS) ./$(TARGET) $(NY) $(NX)

.PHONY: all clean run
