## Checkpoint

### Work Done
Till now, we have understood the naive nfa.c code written by Russ Cox. We had a make a few changes to accomodate our requirements, specifically, the code was not matching whole words like grep does. So we had to modify the code to do that. Also, file handling was not in place, which we fixed. We got a sequential basic nfa regex matcher code working, similar to basic grep functionality. We tried running the program and compare it with how grep performs on a few datasets of different types of workload.

Next, we took a bit of time to understand how OpenCL works and looked at a few examples by running them on our laptops, and migrated the code to OpenCL, where we got a sequential code to work (Essentially a C program), without the use of any GPUs yet. We are right now in the process of parallelizing regex match for given input strings across the input lines in a file by using the GPUs, which we presume will take a bit of time, since we'll have to change the nfa code quite a bit and break up the code into kernels and handle data transfer between the host and the device.

### How we're doing
We think we are on track to deliver what we thought of before, which is parallelising regex match across lines in each file. We are not very sure of if we'll be able to parallelize across directories yet, but it'll be a nice to have. Other extensions like handling backreferences, escape sequences, submatch extraction etc will probably be stretched goals, as our idea here is to not make a better regex matcher in terms of the power, but a better regex matcher in terms of the speed of matching. So, we plan to focus more on trying out different things like trying to exploit SIMD (although this may theoritically not be a good fit for this problem due to divergence), CPU vs GPU, CPU and GPU together etc.

### What we plan to show
We plan to mainly show graphs depicting performance of our OpenCL version against the sequential baseline of the nfa implementation, as well as against a general purpose regex matcher like grep. Also, we plan to have graphs showing which part of the computation takes more time etc.

### Preliminary Results

### Concerns
One concern we have is that OpenCL is not installed on either the andrew machines, or GHC machines, or latedays cluster. It'd be great if we can leverage the Xeon Phi's on the latedays cluster using OpenCL. Apart from that we think it's a matter of getting the logic right and coding it out.

