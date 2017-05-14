#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#define MAX_NUMBER_OF_LINES 20000000
#define THREADS_PER_BLOCK 256
#define NUMBER_OF_BLOCKS 80 

#include "CycleTimer.h"

extern float toBW(int bytes, float sec);

__device__ __inline__ char*
re2post(char *re, int regex_length)
{
    int nalt, natom;
    static char buf[8000];
    char *dst;
    struct {
        int nalt;
        int natom;
    } paren[100], *p;

    p = paren;
    dst = buf;
    nalt = 0;
    natom = 0;
    if(regex_length >= sizeof buf/2)
        return NULL;
    for(; *re; re++){
        switch(*re){
            case '(':
                if(natom > 1){
                    --natom;
                    *dst++ = '.';
                }
                if(p >= paren+100)
                    return NULL;
                p->nalt = nalt;
                p->natom = natom;
                p++;
                nalt = 0;
                natom = 0;
                break;
            case '|':
                if(natom == 0)
                    return NULL;
                while(--natom > 0)
                    *dst++ = '.';
                nalt++;
                break;
            case ')':
                if(p == paren)
                    return NULL;
                if(natom == 0)
                    return NULL;
                while(--natom > 0)
                    *dst++ = '.';
                for(; nalt > 0; nalt--)
                    *dst++ = '|';
                --p;
                nalt = p->nalt;
                natom = p->natom;
                natom++;
                break;
            case '*':
            case '+':
            case '?':
                if(natom == 0)
                    return NULL;
                *dst++ = *re;
                break;
            default:
                if(natom > 1){
                    --natom;
                    *dst++ = '.';
                }
                *dst++ = *re;
                natom++;
                break;
        }
    }
    if(p != paren)
        return NULL;
    while(--natom > 0)
        *dst++ = '.';
    for(; nalt > 0; nalt--)
        *dst++ = '|';
    *dst = 0;
    return buf;
}


/*
 * Represents an NFA state plus zero or one or two arrows exiting.
 * if c == Match, no arrows out; matching state.
 * If c == Split, unlabeled arrows to out and out1 (if != NULL).
 * If c < 256, labeled arrow with character c to out.
 */
enum
{
    Match = 256,
    Split = 257
};
typedef struct State State;
struct State
{
    int c;
    State *out;
    State *out1;
    int lastlist;
};
__constant__ State matchstate = { Match };   /* matching state */
__shared__ int nstate;

__shared__ State states[500];

/* Allocate and initialize State */
    __device__ __inline__ State*
state(int c, State *out, State *out1)
{
    //printf("entering state function\n");
    State *s = &states[nstate];

    nstate++;
    //s = malloc(sizeof *s);
    s->lastlist = 0;
    s->c = c;
    s->out = out;
    s->out1 = out1;
   // printf("Adding new state : %c\n", s->c);
    return s;
}

/*
 * A partially built NFA without the matching state filled in.
 * Frag.start points at the start state.
 * Frag.out is a list of places that need to be set to the
 * next state for this fragment.
 */
typedef struct Frag Frag;
typedef union Ptrlist Ptrlist;
struct Frag
{
    State *start;
    Ptrlist *out;
};

/* Initialize Frag struct. */
    __device__ __inline__ Frag
frag(State *start, Ptrlist *out)
{
    Frag n = { start, out };
    return n;
}

/*
 * Since the out pointers in the list are always 
 * uninitialized, we use the pointers themselves
 * as storage for the Ptrlists.
 */
union Ptrlist
{
    Ptrlist *next;
    State *s;
};

/* Create singleton list containing just outp. */
    __device__ __inline__ Ptrlist*
list1(State **outp)
{
    Ptrlist *l;

    l = (Ptrlist*)outp;
    l->next = NULL;
    return l;
}

/* Patch the list of states at out to point to start. */
    __device__ __inline__ void
patch(Ptrlist *l, State *s)
{
    Ptrlist *next;

    for(; l; l=next){
        next = l->next;
        l->s = s;
    }
}

/* Join the two lists l1 and l2, returning the combination. */
    __device__ __inline__ Ptrlist*
append(Ptrlist *l1, Ptrlist *l2)
{
    Ptrlist *oldl1;

    oldl1 = l1;
    while(l1->next)
        l1 = l1->next;
    l1->next = l2;
    return oldl1;
}

/*
 * Convert postfix regular expression to NFA.
 * Return start state.
 */
    __device__ __inline__ State*
post2nfa(char *postfix)
{
    char *p;
    Frag stack[1000], *stackp, e1, e2, e;
    State *s;

    //fprintf(stderr, "postfix: %s\n", postfix);

    if(postfix == NULL)
        return NULL;

#define push(s) *stackp++ = s
#define pop() *--stackp

    stackp = stack;
    for(p=postfix; *p; p++){
        switch(*p){
            default:
                s = state(*p, NULL, NULL);
                push(frag(s, list1(&s->out)));
                break;
            case '.':   /* catenate */
                e2 = pop();
                e1 = pop();
                patch(e1.out, e2.start);
                push(frag(e1.start, e2.out));
                break;
            case '|':   /* alternate */
                e2 = pop();
                e1 = pop();
                s = state(Split, e1.start, e2.start);
                push(frag(s, append(e1.out, e2.out)));
                break;
            case '?':   /* zero or one */
                e = pop();
                s = state(Split, e.start, NULL);
                push(frag(s, append(e.out, list1(&s->out1))));
                break;
            case '*':   /* zero or more */
                e = pop();
                s = state(Split, e.start, NULL);
                patch(e.out, s);
                push(frag(s, list1(&s->out1)));
                break;
            case '+':   /* one or more */
                e = pop();
                s = state(Split, e.start, NULL);
                patch(e.out, s);
                push(frag(e.start, list1(&s->out1)));
                break;
        }
    }

    e = pop();
    if(stackp != stack)
        return NULL;

    patch(e.out, &matchstate);
    return e.start;
#undef pop
#undef push
}


typedef struct List List;
struct List
{
    State *s[500];
    int n;
};
//__shared__ int listid;

__device__ void custom_addstate(List*, State*);
__device__ __inline__ void step(List*, int, List*);

/* Compute initial state list */
__device__ __inline__ List*
startlist(State *start, List *l)
{
    l->n = 0;
  //  listid++;
    custom_addstate(l, start);
    return l;
}

/* Reset to initial state list */
/*    void
restartlist(List *l)
{
    //printf("Resetting\n");
    l->n = 0;
    custom_addstate(l, start_state);
}
*/
/* Check whether state list contains a match. */
 __device__ __inline__    int
ismatch(List *l)
{
    int i;
    for(i=0; i<l->n; i++)
        if(l->s[i]->c == Match) {
        //    printf("Inside ismatch() ret 1\n");
            return 1;
        }
  //  printf("Inside ismatch() returning 0\n");
    return 0;
}


/* Add s to l, following unlabeled arrows. */
__device__ void
custom_addstate(List *l, State *s)
{
    List state_stack;

    state_stack.n = 0;
    #define push_to_stack(ss) state_stack.s[state_stack.n++] = ss
    #define pop_stack() state_stack.s[--(state_stack.n)]
    #define is_stack_empty() (state_stack.n == 0)

    push_to_stack(s);
    while(!is_stack_empty()) {
        s = pop_stack();
        if(s == NULL)
            break;
        // Surprisingly lastlist is not needed 
    //    s->lastlist = listid;
        if(s->c == Split){
            // follow unlabeled arrows 
            push_to_stack(s->out);
            push_to_stack(s->out1);
        } else {
            l->s[l->n++] = s;
        }
    }
    #undef push_to_stack
    #undef pop_stack
    #undef is_stack_empty
}

/*
 * Step the NFA from the states in clist
 * past the character c,
 * to create next NFA state set nlist.
 */
 __device__ __inline__    void
step(List *clist, int c, List *nlist)
{
    int i;
    State *s;
//    printf("Inside step\n");
  //  listid++;
    nlist->n = 0;
    //printf("Inside step. Evaluating %d\n", c);
    for(i=0; i<clist->n; i++){
        s = clist->s[i];
        //printf("State %d value : %d\n", i, s->c);
        if(s->c == c) {
            //printf("Adding state\n");
            custom_addstate(nlist, s->out);
        }/* else {
        // testing code
        //printf("Lastlist : %d, listid : %d\n", s->lastlist, listid);
        //printf("Character value : %d\n", s->c);
        if (s->lastlist == (listid -1) && s->c == Match) {
        //printf("Adding matchstate\n");
        custom_addstate(nlist, &matchstate);
        } else {
        // Remove everything from list and start over
        restartlist(nlist);
        break;
        }
        }*/
    }
}

/* Run NFA to determine whether it matches s. */
 __device__ __inline__    int
match(State *start, char *s, List *l1, List *l2)
{
    int c;
    List *clist, *nlist, *t;

    clist = startlist(start, l1);
    nlist = l2;
   // printf("After startlist\n");
    for(; *s; s++){
        c = *s & 0xFF;
        step(clist, c, nlist);
        t = clist; clist = nlist; nlist = t;    /* swap clist, nlist */
        if (ismatch(clist)) {
            return 1;
        }
    }
  //  printf("Before ismatch\n");
    return ismatch(clist);
}

__shared__ char* post;
__shared__ State *start;
__global__ void match_kernel(char* regex, int regex_length, char* search_string, int search_string_length,
                    int *linesizes, int number_of_lines) {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;       

    if(index > number_of_lines-1)
        return;

    //printf("Number of lines %d", number_of_lines);
    if (threadIdx.x == 0) {
       // post = re2post(my_pattern, regex_length);
        post = re2post(regex, regex_length);
    //    printf("Regex: %s", regex);
    //    printf("Post: %s", post);
        start = post2nfa(post);
    }

    int offset, end_offset, batchID = 0, batchOffset = 0;
    
    int BatchSize = NUMBER_OF_BLOCKS*THREADS_PER_BLOCK;

    __syncthreads(); 
    for(batchID = 0; (batchOffset + threadIdx.x) < number_of_lines; batchID++)
    {
        char my_search_string[120];
        int i = 0;
        if(index == 0 && batchID == 0)
            offset = 0;
        else
            offset = linesizes[index-1 + batchOffset];
        end_offset = linesizes[index+batchOffset];
        
        for(int k = offset; k < end_offset; k++)
        {
            my_search_string[i] = search_string[k];
            i++;
        }
        my_search_string[i] = '\0';
        
        for(i = 0; my_search_string[i] != '\0'; i++)
        {
           List l1, l2;
           if(match(start, my_search_string + i, &l1, &l2)) {
                printf("%s", my_search_string );
                break;
           }
        }
        batchOffset += BatchSize;
    }
}

void regexMatchCuda(char* regex, int regex_length, char* search_string, 
                int search_string_length, int *linesizes, int number_of_lines) {    

    // compute number of blocks and threads per block
    const int threadsPerBlock = THREADS_PER_BLOCK;
    const int blocks = NUMBER_OF_BLOCKS;

    char* device_regex;
    char* device_search_string;
    int* device_linesizes;
   
    //
    // TODO: allocate device memory buffers on the GPU using
    // cudaMalloc.  The started code issues warnings on build because
    // these buffers are used in the call to saxpy_kernel below
    // without being initialized.
    //
    double startTime = CycleTimer::currentSeconds();
    cudaMalloc(&device_regex, sizeof(char)*(regex_length+1));
    
    double startTime2 = CycleTimer::currentSeconds();
    cudaMalloc(&device_search_string, sizeof(float)*(search_string_length+1));
    cudaMalloc(&device_linesizes, sizeof(int)*(MAX_NUMBER_OF_LINES));

    // start timing after allocation of device memory.

    //
    // TODO: copy input arrays to the GPU using cudaMemcpy
    //
    cudaMemcpy(device_regex, regex, sizeof(char)*(regex_length+1), cudaMemcpyHostToDevice);
    cudaMemcpy(device_search_string, search_string, sizeof(char)*(search_string_length+1), cudaMemcpyHostToDevice);
    cudaMemcpy(device_linesizes, linesizes, sizeof(int)*MAX_NUMBER_OF_LINES, cudaMemcpyHostToDevice);

    double kernelStartTime = CycleTimer::currentSeconds();

    // run match_kernel on the GPU
    match_kernel<<<blocks, threadsPerBlock>>>(device_regex, regex_length, device_search_string, 
                            search_string_length, device_linesizes, number_of_lines);

    //
    // TODO: insert timer here to time only the kernel.  Since the
    // kernel will run asynchronously with the calling CPU thread, you
    // need to call cudaThreadSynchronize() before your timer to
    // ensure the kernel running on the GPU has completed.  (Otherwise
    // you will incorrectly observe that almost no time elapses!)
    //
    cudaThreadSynchronize();


    double kernelEndTime = CycleTimer::currentSeconds();
    //
    // TODO: copy result from GPU using cudaMemcpy
//   
    //cudaMemcpy(result, device_result, sizeof(int), cudaMemcpyDeviceToHost);

    // end timing after result has been copied back into host memory.
    // The time elapsed between startTime and endTime is the total
    // time to copy data to the GPU, run the kernel, and copy the
    // result back to the CPU
    double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }
    
     int totalBytes = 0;
     double overallDuration = endTime - startTime;
     double kernelDuration = kernelEndTime - kernelStartTime;
     double transferDuration = kernelStartTime - startTime + endTime - kernelEndTime;
     double firstMallocDuration = startTime2 - startTime;
     //printf("\nOverall time: %.3f ms\n", 1000.f * overallDuration);
    // printf("Kernel time: %.3f ms\n", 1000.f * kernelDuration);
    // printf("Non-kernel (Memcpy+allMalloc) time: %.3f ms\n", 1000.f * transferDuration);
    // printf("FirstMalloc time: %.3f ms\n", 1000.f * firstMallocDuration);
     printf("\nOverall time:                     %.3f s\n", overallDuration);
     printf("Kernel time:                        %.3f s\n", kernelDuration);
     printf("Non-kernel (Memcpy+allMalloc) time: %.3f s\n", transferDuration);
     printf("FirstMalloc time:                   %.3f s\n", firstMallocDuration);

    //
    // TODO free memory buffers on the GPU
    //
    cudaFree(device_regex);
    cudaFree(device_search_string);
//    cudaFree(device_result);
}

void
printCudaInfo() {

    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
                static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
