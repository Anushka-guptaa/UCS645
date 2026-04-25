# UCS645 - Assignment 3: Parallel Vector Correlation

**Submitted by:** Anushka Gupta  
**Roll No:** 102303358  
**Course:** Parallel and Distributed Computing (UCS645)

## Files Submitted
- `Makefile` – Build system with OpenMP support
- `main.cpp` – Takes `ny` (rows) and `nx` (columns) from command line
- `correlate.cpp` – Implementation of required function `void correlate(...)`

## Features Implemented
- Sequential baseline version (Part 1)
- Parallel version using **OpenMP** (Part 2)
- Pre-computation of row sums and sum-of-squares for better performance
- Command line input for matrix size
- Proper Makefile with `all`, `clean`, `run` targets

## How to Build and Run
```bash
make
make run NY=400 NX=1000 THREADS=4
