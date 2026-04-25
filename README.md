# Parallel and Distributed Computing - Assignment 2

**Submitted by:** Anushka Gupta  
**Roll No:** 102303358  
**Submitted to:** Dr. Saif Nalband  
**Course:** UCS645  

---

## 📋 Overview

This repository contains the solutions for **Assignment 2** of the Parallel and Distributed Computing course (UCS645).
The assignment focuses on implementing and evaluating **OpenMP-based parallelization** for three computationally intensive scientific applications:

1. **Molecular Dynamics** – Lennard-Jones potential force calculation (O(N²) pairwise interactions)
2. **Bioinformatics** – Smith-Waterman local DNA sequence alignment
3. **Scientific Computing** – 2D heat diffusion simulation using finite difference method

---

## 📁 Files Included

| File                        | Description                                      | Parallelized Section                  |
|-----------------------------|--------------------------------------------------|---------------------------------------|
| `lj.cpp`                    | Lennard-Jones force & energy computation         | Outer particle loop                   |
| `sw.cpp`                    | Smith-Waterman dynamic programming alignment     | Score matrix filling                  |
| `heat.cpp`                  | 2D heat diffusion stencil computation            | Outer time-step / grid loop           |
| `Assignment_2_102303358.pdf` | Extra report for experiment questions with terminal screenshots & analysis | -    
| `Assignment_2_Questions_Report.pdf` | Full report of assignment questions with terminal screenshots & analysis | -    


---

## 🛠️ How to Build & Run

### Prerequisites
- Linux environment with `g++` and OpenMP support
- `perf` tool (for performance statistics)

### Compilation
```bash
g++ lj.cpp   -fopenmp -O2 -o lj
g++ sw.cpp   -fopenmp -O2 -o sw
g++ heat.cpp -fopenmp -O2 -o heat
