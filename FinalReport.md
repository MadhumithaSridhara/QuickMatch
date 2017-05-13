## QUICKMATCH
We implemented and evaluated Parallel Regular Expression matching on NVIDIA GTX1080 GPU. The implementation parallellizes Regular Expression matching accross lines in a file. Our implementation achieves ~7x speedup over EGREP and ~30x speedup over the optimized sequential implementation.

## Background
Regex matching is used in applications in Log mining, DNA Sequencing and Spam filters etc. It would be useful to improve the performance of these applications as they often run on huge datasets. The parallellizabilty of Regex Matching comes from the following aspects:
* Matching Different Lines in a file can be done completely in parallel
* There are no communication or synchronization overheads in parallellization as there are no dependencies amongst lines
