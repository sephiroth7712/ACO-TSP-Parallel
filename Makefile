# Build all three MMAS variants. Binaries are placed in the repo root so the
# scripts in scripts/ (which call ./ACO_TSP_*) work when run from here.

CXX       = g++
MPICXX    = mpicxx
CXXFLAGS  = -std=gnu++11 -g
OMPFLAGS  = -fopenmp

BINS = ACO_TSP_Serial ACO_TSP_Threaded ACO_TSP_MPI

.PHONY: all clean

all: $(BINS)

ACO_TSP_Serial: src/ACO_TSP_Serial.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

ACO_TSP_Threaded: src/ACO_TSP_Threaded.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

ACO_TSP_MPI: src/ACO_TSP_MPI.cpp
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

clean:
	rm -f $(BINS)
