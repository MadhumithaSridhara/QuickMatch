## Parallel Regular expression matching
This project implements a fast parallel regular expression matching using OpenCL. Regular expression matching on a large number of files is an embarassingly parallel operation. In this project we explore parallellism across files as well as parallellism within files. Our implementation is based on running parallel instances of a regex engine based on Thompson's NFA. This will use OpenCL to offload the regex matching to a GPU.

### Background
Regular expression matching is used extensively in string-search, 'find' and 'find-and-replace' algorithms. 


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




For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/MadhumithaSridhara/15-418-Parallel-Project/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
