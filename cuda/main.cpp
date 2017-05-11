#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <string.h>
#include <fcntl.h>	
#include <sys/stat.h>	
#include "CycleTimer.h"
#define MAX_NUMBER_OF_LINES 100000

#define THREADS_PER_BLOCK 512
#define NUMBER_OF_BLOCKS 40
void regexMatchCuda(char* regex, int regex_length, char* search_string, 
                int search_string_length, int *linesizes, int number_of_lines,  int* result);   
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
static void
CloseTextFile(
    FILE* fh)
{
    fclose(fh);
}

static unsigned long
ReadFromTextFile(
    FILE*fh, char* buffer, size_t buffer_size)
{
    if(!fh)
        return 0;
    
    unsigned long count = (unsigned long)fread(buffer, buffer_size, 1, fh);
    buffer[buffer_size] = '\0';
    return count;
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
    //count++;
    size_t length;
    ssize_t nread;
    while((nread = getline(&line, &length, fh)) != -1)
    {
        if(count != 0)
           linesizes[count] = linesizes[count-1] + nread;
        else
            linesizes[0] = nread;
//        printf("\n%d Length = %d", count-1, nread);
        count++;
    }
    free(line);
    fclose(fh);
//    printf("\nNumber of lines read = %d", count);
    return count;
}
static char *
LoadTextFromFile( const char *filename, unsigned long *size /* returned file size in bytes */)
{
    FILE* fh = OpenTextFile(filename, size);
    unsigned long bytes = (*size);
    char *text = (char*)malloc(bytes + 1);
    if(!text)
        return 0;
    ReadFromTextFile(fh, text, bytes);
    CloseTextFile(fh);
    return text;
}
int main(int argc, char** argv)
{
    if(argc < 3)
    {
        printf("Usage: ./matchCuda <patternfilename> <searchfilename>");
    }
    unsigned long* results = 0;         // number of patterns found in search buffer
    //unsigned long results_count = 6;    // total number of results expected 
    char* pattern_string = 0;           // pattern string in host memory
    unsigned long pattern_length = 0;   // length of pattern string (in bytes)

    char* search_string = 0;            // search string in host memory
    unsigned long search_length = 0;    // length of search string (in bytes)
    
    int *linesizes;                     // Array of offsets of new lines in file buffer
    
    double loadStartTime = CycleTimer::currentSeconds();
    linesizes = (int *) calloc(MAX_NUMBER_OF_LINES, sizeof(int));

    pattern_string = argv[1];
    pattern_length = strlen(pattern_string);
    if (pattern_string == NULL || pattern_length < 1)
    {
        printf("Reading pattern failed\n");
        return -1;
    }
    search_string = LoadTextFromFile(argv[2], &search_length);
    if (search_string == NULL || search_length < 1)
    {
        printf("Reading search failed\n");
        return -1;
    }
   
    int number_of_lines_read = ReadLineByLineFromTextFile(argv[2], &search_length,  linesizes ); 
    double loadEndTime = CycleTimer::currentSeconds();
    
    int result;
    regexMatchCuda(pattern_string, pattern_length ,search_string, search_length, linesizes, number_of_lines_read, &result);

    printf("Data Processing time %.3f ms\t\n", 1000.f * (loadEndTime-loadStartTime));
    return 0;
}
