## Parallel Regular expression matching
QuickMatch implements a fast parallel regular expression matching using OpenCL. Regular expression matching on a large number of files is an embarassingly parallel operation. In this project we explore parallellism in matching regular expressions across files as well as parallellism in searching for expressions within files. Our implementation is based on running parallel instances of a regex engine based on Thompson's NFA. QuickMatch uses OpenCL to offload the regex matching to a GPU.

### Background
Regular expression matching is used extensively in string-search, 'find' and 'find-and-replace' algorithms. Regular expression matching can be used at a very large scale especially in searching through large sized log and report files. Such applications will benefit greatly from QuickMatch.


### Challenges
- SIMD divergence
- Host and device memory challenges
- Parallelizing generation of NFA
- Managing NFA divergence
- Sorting

### Resources

### Goals and deliverables
- A basic application in OpenCL that can perform parallel regular application matching on a bunch of textfiles
- Performance analysis and comparison with sequential implementation
If time permits, we will also include a comparison with other approaches such as word count. However, we feel word count approach is not always feasible as it can be expensive to create and maintain live index files for all data.


### Platform choice

### Schedule
- Understand NFA
- Sequential implementation in openCL
- Coarse grained parallel implementation by dividing dataset into N partitions
- Fine grained parallel implementation by breaking up lines
- Fine tuning/ prepare presentation


### TEAM
Bharath Kumar M J
Madhumitha Sridhara
