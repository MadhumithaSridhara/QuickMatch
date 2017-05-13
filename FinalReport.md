## QUICKMATCH
We implemented and evaluated Parallel Regular Expression matching on NVIDIA GTX1080 GPU. The implementation parallellizes Regular Expression matching accross lines in a file. Our implementation achieves ~7x speedup over EGREP and ~30x speedup over the optimized sequential implementation.

## Background
### Applications and Motivation
Regex matching is used in applications in Log mining, DNA Sequencing and Spam filters etc. It would be useful to improve the performance of these applications as they often run on huge datasets. The parallellizabilty of Regex Matching comes from the following aspects:
* Matching Different Lines in a file can be done completely in parallel
* There are no communication or synchronization overheads in parallellization as there are no dependencies between different lines

### Algorithm
QuickMatch regex matching is based on Thompson's Non-deterministic Finite Automaton (NFA). This algorithm constructs a finite state machine given a regex. The data to be matched is run through this state machine character by character, and the final state reached determines whether or not the string matched the Regex.

<<INSERT PIC Here>>

The starter code for QuickMatch was taken from Russ Cox's implementation of Thompson's NFA Construction [https://swtch.com/~rsc/regexp/]. The implementation is as follows:
Input: A regular expression and search file
Output: Lines with a matching regex pattern in them are printed to stdout

* Read the file into a buffer called 'search_string' (char *)
* Find the start offsets of different lines in the buffer by reading the file line by line. Save the offsets into an array called 'linesizes' (int *)
* Call the kernel with search_string, linesizes and the lengths of both these arrays

Match Kernel Implementation:
* In each cuda thread block, one thread (threadIdx.x == 0) constructs the NFA from the Regex and saves it in a block __shared__ state. 
* Once the NFA is ready (read __syncThreads()), every thread reads from the search string at an offset determined by its global index (blockIdx.x * blockDim.x + threadIdx) and the linesizes array. i.e Thread with global index 0 will read at offset 0 till the length of the first line which is given by 'linesizes[0]'. Thread with global index 1 reads from linesizes[0] till linesizes[1] and so on. The substring of the search string that represents a line is copied to a thread local array
* The matching is performed on this substring and the NFA in the shared state.
However, for completeness and accuracy, threads process lines in batches. Batchsize = NUMBER_OF_BLOCKS*NUMBER of THREADS per Block. 

Note: The match implemented in Russ Cox's algorithm only matches from start of line. Thus the match function has to be called multiple times with different start positions of the line so that the matching is done for the whole line.

Data Structures:
The NFA itself is a complex datastructure with many pointer indirections. It is represented as a graph using linkedLists
<<ELaborate here>>

The 
### Analysis


