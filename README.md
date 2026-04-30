# High-Performance Grid-Based Environmental Spread Modeling  
### Using Multithreaded CPU Parallelism (OpenMP)

##  Overview
This project implements a **grid-based environmental spread simulation** (forest fire model) using **C++ and OpenMP**. The simulation demonstrates how parallel computing can significantly improve performance for computationally intensive grid-based problems.

The system models fire propagation across a 2D grid where each cell evolves based on its neighbors over discrete time steps.

---

##  Objectives
- Implement a **sequential baseline** simulation
- Develop an **OpenMP parallel version**
- Compare execution time across:
  - Grid sizes: `200×200`, `400×400`, `600×600`
  - Thread counts: `1, 2, 4, 8`
- Analyse:
  - Speedup
  - Parallel efficiency
  - Scalability

---

##  Tech Stack
- **Language:** C++17  
- **Parallelism:** OpenMP  
- **Timing:** std::chrono (high resolution clock)  
- **Compiler:** g++ with `-fopenmp`  
- **Environment:** Kali Linux  

---

## Execution Environment
- Implemented and executed on **Kali Linux VM**
- CPU utilization and thread activity verified using **htop**
- Peak CPU usage reached ~84.9%, confirming effective parallel execution across multiple cores :contentReference[oaicite:1]{index=1}

---

##  Simulation Model

### Cell States:
- `0 → EMPTY`
- `1 → TREE`
- `2 → BURNING`

### Update Rules:
1. Burning → Empty  
2. Tree + Burning neighbor → Burning  
3. Tree (no burning neighbor) → Tree  
4. Empty → Empty  

- Fire starts at **top-center of the grid**
- Simulation runs for **200 steps**
---

## Key Observations
Maximum speedup achieved: ~3.47× (600×600, 8 threads)
Efficiency decreases as thread count increases (Amdahl’s Law)
Larger workloads improve parallel efficiency
No race conditions due to independent cell updates

## Parallelization Strategy
- OpenMP directive used:
  ```cpp
  #pragma omp parallel for collapse(2) schedule(static)
