## [Project Checkpoint](https://madhumithasridhara.github.io/QuickMatch/checkpoint)
## Project Proposal
### Overview
QuickMatch implements a fast parallel regular expression matching using OpenCL. Regular expression matching on a large number of files is an embarassingly parallel operation. In this project we explore parallellism in matching regular expressions across files as well as parallellism in searching for expressions within files. Our implementation is based on running parallel instances of a regex engine based on Thompson's NFA. QuickMatch uses OpenCL to offload the regex matching to a GPU.

### Background
Regular expression matching is used extensively in string-search, 'find' and 'find-and-replace' algorithms. Regular expression matching can be used at a very large scale especially in searching through large sized log and report files. Such applications will benefit greatly from QuickMatch. Thompson's construction algorithm converts a regular expression into a Nondeterministic Finite Automaton(NFA)or a finite state machine(FSM). Finding a pattern in large files is compute intensive, and we are trying to explore if GPUs can be exploited for this purpose. One aspect of parallelism is working on different lines in parallel, other is to work on different files. Apart from this, in order to use Thompson's NFA based algorithm, the NFA has to be constructed from the given regular expression. There is benefit in doing this faster. We want to try and explore if this can even be done. The below is a basic high level pseudo code of the algorithm : 

~~~~
Load the (directory/file), (regular expression)
Construct NFA from the regular expression (Try parallelizing)
if (directory) {
   foreach file in directory (in parallel) {
      parallel_regex_search();
   }
} else {
   parallel_regex_search();
}

parallel_regex_search() {
   foreach line in file (in parallel) {
      if (match(regex)) {
         print regex
      }
   }
}
~~~~


### Challenges
- One of the biggest challenges of using a GPU would come in the form of memory management. Performance will vary greatly depending on whether or not the entire working set of files will fit in the GPU memory. Also, a significant amount of time will be wasted on transferring to GPU memory from host memory and vice-versa.
- If SIMD Vectors are used to get speedup for fine-grained parallelism accross lines of text in files, there will be a serious issue of SIMD divergence as some vector lanes which find a match quickly will exit much before those which have to search for the expression till a new line character.
- Since NFA is non deterministic, there are many ways to handle the situation where there are multiple next states (see [2]). One of the suggested ways is to maintain state for all possible paths and drop a path which ends in failure. This approach parallel paths in a thread of execution and has to be handled properly.
- One of the rate limiting steps could be the generation of the NFA engine itself. Since the NFA engine depends on the input regex it can only be created at runtime. Parallellization as well as latency hiding (Hide the latency of host-to-memory transfer with this engine) techniques can be attempted to overcome this challenge 
- Since the regex matching will happen in parallel and out of order, the results will not appear in the order of files searched, this is not ideal for users. It might be useful to reorder and regroup the output according to the files. 

### Resources
We plan to use the Xeon Phis available to us in the intel cluster to test our code. The main resources we'll be using are these two:
- NFA paper by Thompson\[[1](https://dx.doi.org/10.1145%2F363347.363387)\]
- Russ Cox's implementation of the Thompson NFA in C\[[2](https://swtch.com/~rsc/regexp/nfa.c.txt)\]
- Russ Cox's excellent blog on regular expressions\[[3](https://swtch.com/~rsc/regexp/regexp1.html)\]

### Goals and deliverables
- A basic application in OpenCL that can perform parallel regular application matching on a bunch of textfiles
- Performance analysis and comparison with sequential implementation
- If time permits, we will also include a comparison with other approaches such as word count. However, we feel word count approach is not always feasible as it can be expensive to create and maintain live index files for all data.


### Platform choice
We plan to use the Intel Xeon Phis available to us provided by Intel, which we used in assignment3. We want to test our code mainly on this.

### Schedule
* Week 1: Understand Thompson's NFA algorithm for regex matching
    * *What we should do*: We plan to use the starter code provided in Russ Cox's website [2] and understand how the sequential implementation works. We will also find sample input files which can serve as benchmarks for performance.
    * *What we actually did*: Figured out that there are some bugs in the starter code and it doesn't always work the way we need it to. Making fixes to the starter code for it to work like Grep.
* Week 2: Sequential implementation in openCL
    * *What we should do*: We will port the code to OpenCL and run the sequential version of the code on the GPU. This step will also help us figure out how the input and output transfers will need to be done.
    * *What we actually did*: Ran example codes on OpenCL on personal laptops and started porting the NFA code to OpenCL.
    
* Week 3: Coarse grained parallel implementation 
   * *What we should do*
   1. First Half: Complete porting the NFA code to OpenCL and modify to allow single threaded version to "dispatch" to device using OpenCL
   2. Second Half: Start parallellizing the code by breaking up the data into partitions to be read by different instances of the kernel
   
   * *What we actually did*: 
* Week 4: Fine grained parallel implementation
   * *What we should do*: This week we will use SIMD to get better performance on QuickMatch 
    1. First Half: Complete the parallel code using OpenCL on one platform (MacBook? with Intel Integrated HD graphics )
    2. Second Half: Get a working version of the code on a different platform (latedays Xeon Phi?) and try optimizing it
    
    * *What we actually did*: 
* Week 5: Fine tuning/ prepare presentation
    * *What we should do*: Run benchmarks, collect results and look for scopes for improvement. Also prepare for the final demo and presentation. This week will also serve as a buffer in case any of the steps take longer than expected.
    1. First Half: Run QuickMatch with different datasets / different architectures and observe performance. Try to improve any common bottlenecks and characterize the memory/compute usage.
    2. Second Half: Buffer and prepare for the presentation.
    * *What we actually did*: 

### References
1. [Thompson, Ken. "Programming techniques: Regular expression search algorithm." Communications of the ACM 11.6 (1968): 419-422.](https://dx.doi.org/10.1145%2F363347.363387)
2. [Implementing Regular Expressions. N.p., n.d. Web.](https://swtch.com/~rsc/regexp/regexp1.html)

### TEAM
Bharath Kumar M J(bjaganna) & Madhumitha Sridhara(madhumit)
