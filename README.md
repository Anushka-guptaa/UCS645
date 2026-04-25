# UCS645: Parallel & Distributed Computing  
**Assignment 5: MPI Part II (Blocking vs Non-Blocking Communication)**

---

## Overview
This repository contains the complete solutions for all **5 questions** of Assignment 5.  
The programs demonstrate:
- Blocking vs Non-blocking communication
- Collective operations (`MPI_Bcast`, `MPI_Reduce`)
- Performance measurement with `MPI_Wtime()`
- Master-Slave dynamic load balancing

---

## 🗂 Files Included

| Question | File                    | Description                                      |
|----------|-------------------------|--------------------------------------------------|
| 1        | `q1_daxpy.c`            | Parallel DAXPY operation + speedup measurement   |
| 2        | `q2_broadcast.c`        | MyBcast (linear) vs optimized MPI_Bcast          |
| 3        | `q3_dotproduct.c`       | Distributed Dot Product + Amdahl’s Law analysis  |
| 4        | `q4_primes.c`           | Master-Slave Prime Number Finder                 |
| 5        | `q5_perfect.c`          | Master-Slave Perfect Number Finder               |
| -        | `Makefile`              | Easy compilation and execution                   |

---

## How to Compile and Run

### Using Makefile 
```bash
make          # Compile all 5 programs
make run1     # Run Question 1
make run2     # Run Question 2
make run3     # Run Question 3
make run4     # Run Question 4
make run5     # Run Question 5

Manual Commands
Bashmpicc -o qX_xxx qX_xxx.c
mpirun --allow-run-as-root --oversubscribe -np 4 ./qX_xxx

Key Results (Tested on Google Colab)

Q2 Broadcast Race: MPI_Bcast (0.078s) is ~4.5× faster than manual linear MyBcast (0.35s)
Q3 Dot Product: Good speedup observed as the number of processes increases
Q4 & Q5: Master-Slave pattern successfully distributes workload dynamically
