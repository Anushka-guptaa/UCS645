CC = mpicc
CFLAGS = -O2 -Wall

all: q1 q2 q3 q4 q5

q1:
	$(CC) $(CFLAGS) -o q1_daxpy q1_daxpy.c

q2:
	$(CC) $(CFLAGS) -o q2_broadcast q2_broadcast.c

q3:
	$(CC) $(CFLAGS) -o q3_dotproduct q3_dotproduct.c

q4:
	$(CC) $(CFLAGS) -o q4_primes q4_primes.c

q5:
	$(CC) $(CFLAGS) -o q5_perfect q5_perfect.c

clean:
	rm -f q1_daxpy q2_broadcast q3_dotproduct q4_primes q5_perfect

run1: q1
	mpirun --allow-run-as-root --oversubscribe -np 4 ./q1_daxpy

run2: q2
	mpirun --allow-run-as-root --oversubscribe -np 4 ./q2_broadcast

run3: q3
	mpirun --allow-run-as-root --oversubscribe -np 4 ./q3_dotproduct

run4: q4
	mpirun --allow-run-as-root --oversubscribe -np 4 ./q4_primes

run5: q5
	mpirun --allow-run-as-root --oversubscribe -np 4 ./q5_perfect
