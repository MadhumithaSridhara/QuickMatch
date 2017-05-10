#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

extern float toBW(int bytes, float sec);

/*
 * Convert infix regexp re to postfix notation.
 * Insert . as explicit concatenation operator.
 * Cheesy parser, return static buffer.
 */
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

__shared__ State states[100];

/* Allocate and initialize State */
    __device__ __inline__ State*
state(int c, State *out, State *out1)
{
    State *s = &states[nstate];

    nstate++;
    //s = malloc(sizeof *s);
    s->lastlist = 0;
    s->c = c;
    s->out = out;
    s->out1 = out1;
    printf("Adding new state : %c\n", s->c);
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
    State *s[100];
    int n;
};
__shared__ List l1, l2;
__shared__ int listid;

__device__ void addstate(List*, State*);
__device__ __inline__ void step(List*, int, List*);

/* Compute initial state list */
__device__ __inline__ List*
startlist(State *start, List *l)
{
    printf("starting startlist\n");
    l->n = 0;
    printf("starting startlist\n");
    listid++;
    printf("starting startlist\n");
    printf("start : %c\n", start->c);
    addstate(l, start);
    printf("exiting startlist\n");
    return l;
}

/* Reset to initial state list */
/*    void
restartlist(List *l)
{
    //printf("Resetting\n");
    l->n = 0;
    addstate(l, start_state);
}
*/
/* Check whether state list contains a match. */
 __device__ __inline__    int
ismatch(List *l)
{
    int i;

    for(i=0; i<l->n; i++)
        if(l->s[i] == &matchstate)
            return 1;
    return 0;
}

/* Add s to l, following unlabeled arrows. */
__device__ void
addstate(List *l, State *s)
{
    printf("starting addstate\n");
    if(s == NULL || s->lastlist == listid)
        return;
    printf("not returning yet addstate\n");
    s->lastlist = listid;
    printf("set lastlist addstate\n");
    if(s->c == Split){
        /* follow unlabeled arrows */
        addstate(l, s->out);
        addstate(l, s->out1);
        return;
    }
    printf("before accessing l addstate\n");
    l->s[l->n++] = s;
    printf("ending addstate\n");
    //printf("Added new state\n");
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

    listid++;
    nlist->n = 0;
    //printf("Inside step. Evaluating %d\n", c);
    for(i=0; i<clist->n; i++){
        s = clist->s[i];
        //printf("State %d value : %d\n", i, s->c);
        if(s->c == c) {
            //printf("Adding state\n");
            addstate(nlist, s->out);
        }/* else {
        // testing code
        //printf("Lastlist : %d, listid : %d\n", s->lastlist, listid);
        //printf("Character value : %d\n", s->c);
        if (s->lastlist == (listid -1) && s->c == Match) {
        //printf("Adding matchstate\n");
        addstate(nlist, &matchstate);
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
match(State *start, char *s)
{
    int i, c;
    List *clist, *nlist, *t;

    clist = startlist(start, &l1);
    nlist = &l2;
    for(; *s; s++){
        //printf("Character : %c\n", *s);
        c = *s & 0xFF;
        step(clist, c, nlist);
        t = clist; clist = nlist; nlist = t;    /* swap clist, nlist */
    }
    return ismatch(clist);
}

__global__ void
match_kernel(char* regex, int regex_length, char* search_string, int search_string_length, int* result) {

    char *post;
    State *start;
    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;       

    if (index == 0) {
        post = re2post(regex, regex_length);
        printf("post : %s\n", post);

        start = post2nfa(post);
        printf("start out : %c\n", start->out->c);
        printf("start out1 : %c\n", start->out1->c);
        if(match(start, search_string)) {
            printf("%s\n", search_string);
            *result = 1;
        }
    }
}

void
regexMatchCuda(char* regex, int regex_length, char* search_string, int search_string_length, int* result) {    

    // compute number of blocks and threads per block
    const int threadsPerBlock = 512;
    const int blocks = 1;

    char* device_regex;
    char* device_search_string;
    int* device_result;

    //
    // TODO: allocate device memory buffers on the GPU using
    // cudaMalloc.  The started code issues warnings on build because
    // these buffers are used in the call to saxpy_kernel below
    // without being initialized.
    //
    cudaMalloc(&device_regex, sizeof(char)*regex_length);
    cudaMalloc(&device_search_string, sizeof(float)*search_string_length);
    cudaMalloc(&device_result, sizeof(int));

    // start timing after allocation of device memory.
    //double startTime = CycleTimer::currentSeconds();

    //
    // TODO: copy input arrays to the GPU using cudaMemcpy
    //
    cudaMemcpy(device_regex, regex, sizeof(char)*regex_length, cudaMemcpyHostToDevice);
    cudaMemcpy(device_search_string, search_string, sizeof(char)*search_string_length, cudaMemcpyHostToDevice);

    //double kernelStartTime = CycleTimer::currentSeconds();

    // run match_kernel on the GPU
    match_kernel<<<blocks, threadsPerBlock>>>(device_regex, regex_length, device_search_string, search_string_length, device_result);

    //
    // TODO: insert timer here to time only the kernel.  Since the
    // kernel will run asynchronously with the calling CPU thread, you
    // need to call cudaThreadSynchronize() before your timer to
    // ensure the kernel running on the GPU has completed.  (Otherwise
    // you will incorrectly observe that almost no time elapses!)
    //
    cudaThreadSynchronize();


    //double kernelEndTime = CycleTimer::currentSeconds();
    //
    // TODO: copy result from GPU using cudaMemcpy
    //
    cudaMemcpy(result, device_result, sizeof(int), cudaMemcpyDeviceToHost);

    // end timing after result has been copied back into host memory.
    // The time elapsed between startTime and endTime is the total
    // time to copy data to the GPU, run the kernel, and copy the
    // result back to the CPU
    //double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    // double overallDuration = endTime - startTime;
    // double kernelDuration = kernelEndTime - kernelStartTime;
    // double transferDuration = kernelStartTime - startTime + endTime - kernelEndTime;
    // printf("Overall time: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW(totalBytes, overallDuration));
    // printf("Kernel time: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * kernelDuration, toBW(totalBytes, kernelDuration));
    // printf("Transfer time: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * transferDuration, toBW(totalBytes, transferDuration));

    //
    // TODO free memory buffers on the GPU
    //
    cudaFree(device_regex);
    cudaFree(device_search_string);
    cudaFree(device_result);
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
