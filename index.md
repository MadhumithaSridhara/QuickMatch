## Project Proposal
QuickMatch implements a fast parallel regular expression matching using OpenCL. Regular expression matching on a large number of files is an embarassingly parallel operation. In this project we explore parallellism in matching regular expressions across files as well as parallellism in searching for expressions within files. Our implementation is based on running parallel instances of a regex engine based on Thompson's NFA. QuickMatch uses OpenCL to offload the regex matching to a GPU.

### Background
Regular expression matching is used extensively in string-search, 'find' and 'find-and-replace' algorithms. Regular expression matching can be used at a very large scale especially in searching through large sized log and report files. Such applications will benefit greatly from QuickMatch. Thompson's construction algorithm converts a regular expression into a Nondeterministic Finite Automaton(NFA)or a finite state machine(FSM).


### Challenges
- One of the biggest challenges of using a GPU would come in the form of memory management. Performance will vary greatly depending on whether or not the entire working set of files will fit in the GPU memory. Also, a significant amount of time will be wasted on transferring to GPU memory from host memory and vice-versa.
- If SIMD Vectors are used to get speedup for fine-grained parallelism accross lines of text in files, there will be a serious issue of SIMD divergence as some vector lanes which find a match quickly will exit much before those which have to search for the expression till a new line character.
- Since NFA is non deterministic, there are many ways to handle the situation where there are multiple next states (see [2]). One of the suggested ways is to maintain state for all possible paths and drop a path which ends in failure. This approach parallel paths in a thread of execution and has to be handled properly.
- One of the rate limiting steps could be the generation of the NFA engine itself. Since the NFA engine depends on the input regex it can only be created at runtime. Parallellization as well as latency hiding (Hide the latency of host-to-memory transfer with this engine) techniques can be attempted to overcome this challenge 
- Since the regex matching will happen in parallel and out of order, the results will not appear in the order of files searched, this is not ideal for users. It might be useful to reorder and regroup the output according to the files. 

### Resources

### Goals and deliverables
- A basic application in OpenCL that can perform parallel regular application matching on a bunch of textfiles
- Performance analysis and comparison with sequential implementation
- If time permits, we will also include a comparison with other approaches such as word count. However, we feel word count approach is not always feasible as it can be expensive to create and maintain live index files for all data.


### Platform choice

### Schedule
- Understand NFA
- Sequential implementation in openCL
- Coarse grained parallel implementation by dividing dataset into N partitions
- Fine grained parallel implementation by breaking up lines
- Fine tuning/ prepare presentation

### References
1. Thompson, Ken. "Programming techniques: Regular expression search algorithm." Communications of the ACM 11.6 (1968): 419-422.
2. "Implementing Regular Expressions." Implementing Regular Expressions. N.p., n.d. Web.[https://swtch.com/~rsc/regexp/regexp1.html] 

### TEAM
Bharath Kumar M J & Madhumitha Sridhara
