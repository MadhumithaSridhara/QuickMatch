#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <string.h>

void regexMatchCuda(char* regex, int regex_length, char* search_string, int search_string_length, int* result);
void printCudaInfo();


// return GB/s
float toBW(int bytes, float sec) {
  return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}


void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -n  --arraysize <INT>  Number of elements in arrays\n");
    printf("  -?  --help             This message\n");
}
/*
static FILE *
OpenTextFile(
    const char* filename, unsigned long* size)
{
    struct stat statbuf;
    FILE *fh;

    fh = fopen(filename, "r");
    if (fh == 0)
        return NULL;

    stat(filename, &statbuf);
    if(size)
        (*size) = (unsigned long)statbuf.st_size;

    return fh;
}

static int
ReadLineByLineFromTextFile(
    const char *filename, unsigned long *size,  int *linesizes)
{
    FILE* fh = OpenTextFile(filename, size);
    
    if(!fh)
        return 0;
    char *line = NULL;
    int count = 0;
    linesizes[count] = 0;
    count++;
    size_t length;
    ssize_t nread;
    while((nread = getline(&line, &length, fh)) != -1)
    {
        linesizes[count] = linesizes[count-1] + nread;
        printf("\n%d Length = %d", count-1, nread);
        count++;
    }
    free(line);
    fclose(fh);
    printf("\nNumber of lines read = %d", count);
    return count;
}
*/
int main(int argc, char** argv)
{


    //int number_of_lines_read = ReadLineByLineFromTextFile(argv[1], &search_length,  linesizes ); 
    char *regex = "aaa";
    char *search_string = "aaaaaa";
    int result;
    regexMatchCuda(regex, strlen(regex), search_string, strlen(search_string), &result);
    // int i;
    // char *post;
    // State *start;

    // if(argc < 3){
    //     fprintf(stderr, "usage: nfa regexp string...\n");
    //     return 1;
    // }

    // post = re2post(argv[1]);
    // if(post == NULL){
    //     fprintf(stderr, "bad regexp %s\n", argv[1]);
    //     return 1;
    // }

    // start = post2nfa(post);
    // if(start == NULL){
    //     fprintf(stderr, "error in post2nfa %s\n", post);
    //     return 1;
    // }
    // start_state = start;
    // l1.s = malloc(nstate*sizeof l1.s[0]);
    // l2.s = malloc(nstate*sizeof l2.s[0]);
    // Adding file input checking line by line

   //  int N = 20 * 1000 * 1000;

   //  const float alpha = 2.0f;
   //  float* xarray = new float[N];
   //  float* yarray = new float[N];
   //  float* resultarray = new float[N];

   //  // load X, Y, store result
   //  for (int i=0; i<N; i++) {
   //      xarray[i] = yarray[i] = i % 10;
   //      resultarray[i] = 0.f;
   // }

   //  printCudaInfo();

   //  for (int i=0; i<3; i++) {
   //    saxpyCuda(N, alpha, xarray, yarray, resultarray);
   //  }

   //  delete [] xarray;
   //  delete [] yarray;
   //  delete [] resultarray;

    return 0;
}
