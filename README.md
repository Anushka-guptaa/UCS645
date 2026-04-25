# UCS645: Parallel & Distributed Computing  
**Assignment 4**

**Submitted by:** Anushka Gupta

---

## 📋 Overview
This repository contains the **complete solutions** for the Laboratory Exercises of Assignment 4 on Message Passing Interface (MPI). All programs are written in C and have been successfully tested on **Google Colab** using MPICH.

## 🗂 Programs Included

| Exercise | File Name              | Description                                      |
|----------|------------------------|--------------------------------------------------|
| 1        | `ring_comm.c`          | Ring Communication (message passing in circle)   |
| 2        | `array_sum.c`          | Parallel Array Sum using Scatter + Reduce        |
| 3        | `max_min.c`            | Global Max & Min using MPI_MAXLOC / MPI_MINLOC   |
| 4        | `dot_product.c`        | Parallel Dot Product of two vectors              |

---

## 🚀 How to Compile and Run

### On Google Colab (Recommended)
```bash
# Compile
!mpicc -o program_name program_name.c

# Run with 4 processes
!mpirun --allow-run-as-root --oversubscribe -np 4 ./program_name
