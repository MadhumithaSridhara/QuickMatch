#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <getopt.h>
#include <time.h>
#include <sys/time.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __APPLE__
    #include <OpenCL/opencl.h>
    #include <mach/mach_time.h>
    typedef uint64_t                    time_delta_t;
    typedef mach_timebase_info_data_t   frequency_t;
#else
    #include <CL/cl.h>
    typedef struct timeval              time_delta_t;
    typedef double                      frequency_t;
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#define Info(...)                   fprintf(stdout, __VA_ARGS__)
#define Warning(...)                fprintf(stderr, __VA_ARGS__)
#define SEPARATOR ("----------------------------------------------------------------------------------------\n")
#define MAX_NUMBER_OF_CHARS_IN_A_LINE 128 
#define MAX_NUMBER_OF_LINES 10 

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* KernelFilename          = "kernel.cl";
const char* KernelMethodName        = "PatternMatcher";
    
////////////////////////////////////////////////////////////////////////////////////////////////////

static void
Usage(const char *progName) 
{
   Info("Usage: %s [options]\n", progName);
   Info("options_t:\n");
   Info("  -h, --help                  This message\n");
   Info("  -q, --quiet                 Disable diagnostic output\n");
   Info("  -c, --cpu                   Use CPU device for executiong (if available).\n");
   Info("  -t, --timing                Gather profiling information\n");
   Info("  -v, --verify                Check the output matches expectations\n");
   Info("  -p, --pattern <FILE>        Filename of text file containing pattern string\n");
   Info("  -s, --search  <FILE>        Filename of text file containing search string\n");
   Info("  -g, --global <N>            Global workgroup size\n");
   Info("  -l, --local <N>             Local workgroup size\n");
   exit(1);
}

typedef struct _options_t 
{
    int verify;
    int cpu;
    int timing;
    int quiet;
    size_t global;
    size_t local;
    size_t iterations;
    char* pattern;
    char* search;
} options_t;

static void
ParseOptions(options_t *opts, int argc, char *argv[]) 
{
    int opt;
    
    static struct option long_options[] = {
        {"help",         0, 0, 'h'},
        {"quiet",        0, 0, 'q'},
        {"cpu   ",       0, 0, 'c'},
        {"timing",       0, 0, 't'},
        {"verify",       0, 0, 'v'},
        {"global",       1, 0, 'g'},
        {"local",        1, 0, 'l'},
        {"iterations",   1, 0, 'i'},
        {"pattern",      required_argument, 0, 'p'},
        {"search",       required_argument, 0, 's'},
    };

    while ((opt = getopt_long(argc, argv, "hqci:g:l:tvp:s:", long_options, NULL)) != EOF) 
    {
        switch(opt) 
        {
        case 'q': opts->quiet = 1; break;
        case 'c': opts->cpu = 1; break;
        case 's': opts->search = strdup(optarg); break;
        case 'p': opts->pattern = strdup(optarg); break;
        case 'g': opts->global = strtoul(optarg, NULL, 0); break;
        case 'l': opts->local = strtoul(optarg, NULL, 0); break;
        case 'i': opts->iterations = strtoul(optarg, NULL, 0); break;
        case 't': opts->timing = 1; break;
        case 'v': opts->verify = 1; break;
        case 'h':
        default:
            Usage(argv[0]);
            break;
        }
    }

    if(!opts->search || !opts->pattern)
    {
        Warning("ERROR: Both pattern and search filenames are required!\n");
        Usage(argv[0]);
    }
   
   return;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static double 
SubtractTimeInSecondsForHost(
    time_delta_t a, time_delta_t b, frequency_t frequency)
{
    double delta;
#ifdef __APPLE__
    time_delta_t difference = a - b;                                    
    delta = 1e-9 * (double) frequency.numer / (double)frequency.denom;  
    delta = delta * difference;                                     
#else
    (void) (frequency);                                                 
    delta = ((double)a.tv_sec + 1.0e-6 * (double)a.tv_usec);    
    delta -= ((double)b.tv_sec + 1.0e-6 * (double)b.tv_usec);   
#endif
    return delta;
}

static frequency_t 
GetTimerFrequencyForHost(void)
{
    frequency_t frequency;
#ifdef __APPLE__
    mach_timebase_info(&(frequency));
#else
    frequency = 1.0;
#endif
    return frequency;
}

static time_delta_t
GetCurrentTimeForHost(void)
{
    time_delta_t current;
#ifdef __APPLE__
    current = mach_absolute_time();
#else
    gettimeofday(&current, NULL);
#endif
    return current;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static const char*
GetErrorString(
    cl_int error)
{
    switch(error)
    {
    case(CL_SUCCESS):                           return "Success";
    case(CL_DEVICE_NOT_FOUND):                  return "Device not found!";
    case(CL_DEVICE_NOT_AVAILABLE):              return "Device not available!";
    case(CL_MEM_OBJECT_ALLOCATION_FAILURE):     return "Memory object allocation failure!";
    case(CL_OUT_OF_RESOURCES):                  return "Out of resources!";
    case(CL_OUT_OF_HOST_MEMORY):                return "Out of host memory!";
    case(CL_PROFILING_INFO_NOT_AVAILABLE):      return "Profiling information not available!";
    case(CL_MEM_COPY_OVERLAP):                  return "Overlap detected in memory copy operation!";
    case(CL_IMAGE_FORMAT_MISMATCH):             return "Image format mismatch detected!";
    case(CL_IMAGE_FORMAT_NOT_SUPPORTED):        return "Image format not supported!";
    case(CL_INVALID_VALUE):                     return "Invalid value!";
    case(CL_INVALID_DEVICE_TYPE):               return "Invalid device type!";
    case(CL_INVALID_DEVICE):                    return "Invalid device!";
    case(CL_INVALID_CONTEXT):                   return "Invalid context!";
    case(CL_INVALID_QUEUE_PROPERTIES):          return "Invalid queue properties!";
    case(CL_INVALID_COMMAND_QUEUE):             return "Invalid command queue!";
    case(CL_INVALID_HOST_PTR):                  return "Invalid host pointer address!";
    case(CL_INVALID_MEM_OBJECT):                return "Invalid memory object!";
    case(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR):   return "Invalid image format descriptor!";
    case(CL_INVALID_IMAGE_SIZE):                return "Invalid image size!";
    case(CL_INVALID_SAMPLER):                   return "Invalid sampler!";
    case(CL_INVALID_BINARY):                    return "Invalid binary!";
    case(CL_INVALID_BUILD_OPTIONS):             return "Invalid build options!";
    case(CL_INVALID_PROGRAM):                   return "Invalid program object!";
    case(CL_INVALID_PROGRAM_EXECUTABLE):        return "Invalid program executable!";
    case(CL_INVALID_KERNEL_NAME):               return "Invalid kernel name!";
    case(CL_INVALID_KERNEL):                    return "Invalid kernel object!";
    case(CL_INVALID_ARG_INDEX):                 return "Invalid index for kernel argument!";
    case(CL_INVALID_ARG_VALUE):                 return "Invalid value for kernel argument!";
    case(CL_INVALID_ARG_SIZE):                  return "Invalid size for kernel argument!";
    case(CL_INVALID_KERNEL_ARGS):               return "Invalid kernel arguments!";
    case(CL_INVALID_WORK_DIMENSION):            return "Invalid work dimension!";
    case(CL_INVALID_WORK_GROUP_SIZE):           return "Invalid work group size!";
    case(CL_INVALID_GLOBAL_OFFSET):             return "Invalid global offset!";
    case(CL_INVALID_EVENT_WAIT_LIST):           return "Invalid event wait list!";
    case(CL_INVALID_EVENT):                     return "Invalid event!";
    case(CL_INVALID_OPERATION):                 return "Invalid operation!";
    case(CL_INVALID_GL_OBJECT):                 return "Invalid OpenGL object!";
    case(CL_INVALID_BUFFER_SIZE):               return "Invalid buffer size!";
    default:                                    return "Unknown error!";
    };
    
    return "Unknown error";
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static FILE *
OpenTextFile(
    const char* filename, unsigned long* size /* returned file size in bytes */)
{
    struct stat statbuf;
    FILE        *fh;

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
    CloseTextFile(fh);
    printf("\nNumber of lines read = %d", count);
    return count;
}
////////////////////////////////////////////////////////////////////////////////////////////////////

static unsigned long
VerifyPatternCountInSearchBuffer(
    char* pattern_string, unsigned long pattern_length,
    char* search_string, unsigned long search_length) 
{
    unsigned long count = 0;
    unsigned long offset = 0;
    char* found = strstr(search_string, pattern_string);
    
    while(found != NULL) 
    {
        offset = found - search_string + 1;
        found = strstr(search_string + offset, pattern_string);
        count++;
    }   
    
    return count;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static char *
LoadTextFromFile(
    const char *filename, unsigned long *size /* returned file size in bytes */)
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

////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
    int i;
    int err;                            // error code returned from api calls
    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation
    int iterations;                     // number of iterations to execute

    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    cl_platform_id platform = 0;        // default system platform
    
    cl_mem pattern_input;               // device memory used for the input pattern string
    cl_mem search_input;                // device memory used for the input search string
    cl_mem results_output;              // device memory used for the output array
    cl_mem line_lengths_input;              // device memory used for the output array
    
    unsigned long* results = 0;         // number of patterns found in search buffer
    unsigned long results_count = 6;    // total number of results expected 
    char* pattern_string = 0;           // pattern string in host memory
    unsigned long pattern_length = 0;   // length of pattern string (in bytes)

    char* search_string = 0;            // search string in host memory
    unsigned long search_length = 0;    // length of search string (in bytes)
    
    int *linesizes;                     // Array of offsets of new lines in file buffer

    linesizes = (int *) calloc(MAX_NUMBER_OF_LINES, sizeof(int));
    frequency_t frequency = {0};        // host timer frequency
    time_delta_t t0 = {0};              // host timer values 
    time_delta_t t1 = {0};  
    
    // Parse the commandline options
    //
    options_t options = {0};
    ParseOptions(&options, argc, argv);
    
    // Load the pattern and search strings into host memory
    //
    if(!options.quiet)
    {
        Info(SEPARATOR);
        Info("Loading pattern string from file '%s'\n", options.pattern);
    }
    
    pattern_string = LoadTextFromFile(options.pattern, &pattern_length);
    if (pattern_string == NULL || pattern_length < 1)
    {
        Warning("ERROR: Failed to load pattern string from file!\n");
        return EXIT_FAILURE;
    }

    if(!options.quiet)
        Info("Loading search string from file '%s'\n", options.search);

    search_string = LoadTextFromFile(options.search, &search_length);
    if (search_string == NULL || search_length < 1)
    {
        Warning("ERROR: Failed to load search string from file!\n");
        return EXIT_FAILURE;
    }
   
    int number_of_lines_read = ReadLineByLineFromTextFile(options.search, &search_length,  linesizes ); 

    results = (unsigned long*)malloc(sizeof(unsigned long) * results_count);
    if(!results)
    {
        Warning("ERROR: Failed to allocate host memory for storing results!\n");
        return EXIT_FAILURE;
    }
    memset(results, 0, sizeof(unsigned long) * results_count);
    
    if(!options.quiet)
    {
        Info(SEPARATOR);
        Info("Locating pattern string '%s' (%lu bytes) in search string '%s' (%lu bytes)\n", 
            options.pattern, (unsigned long)pattern_length, 
            options.search, (unsigned long)search_length);    
    }
    
    // Connect to a compute device
    //
    int gpu = options.cpu ? 0 : 1; // use a GPU device unless told otherwise
    err = clGetDeviceIDs(platform, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        Warning("ERROR: Failed to create a device group! %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    // Report which device we are using
    //
    if(!options.quiet)
    {
        cl_char vendor_name[1024] = {0};
        cl_char device_name[1024] = {0};
        
        err = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, NULL);
        err|= clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
        if (err != CL_SUCCESS)
        {
            Warning("Error: Failed to retrieve device info! %s\n", GetErrorString(err));
            return EXIT_FAILURE;
        }
    
        Info(SEPARATOR);
        Info("Connecting to %s %s...\n", vendor_name, device_name);
    }
  
    // Create a compute context 
    //
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context || err != CL_SUCCESS)
    {
        Warning("ERROR: Failed to create a compute context! %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    // Create a command commands
    //
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands || err != CL_SUCCESS)
    {
        Warning("ERROR: Failed to create a command commands! %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    // Load the kernel source code from disk
    //
    unsigned long kernel_length = 0;
    char* kernel_source = LoadTextFromFile(KernelFilename, &kernel_length);
    if (!kernel_source)
    {
        Warning("ERROR: Failed to load kernel from file!\n");
        return EXIT_FAILURE;
    }
    
    //
    // Create the compute program from the source buffer
    //
    program = clCreateProgramWithSource(context, 1, (const char **) &kernel_source, NULL, &err);
    if (!program || err != CL_SUCCESS)
    {
        Warning("ERROR: Failed to create compute program! %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    // Build the program executable
    //
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        Warning("ERROR: Failed to build program executable! %s\n", GetErrorString(err));
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        Warning("%s\n", buffer);
        return EXIT_FAILURE;
    }

    // Create the compute kernel in the program we wish to run
    //
    kernel = clCreateKernel(program, KernelMethodName, &err);
    if (!kernel || err != CL_SUCCESS)
    {
        Warning("ERROR: Failed to create compute kernel! %s\n", GetErrorString(err));
        exit(1);
    }

    // Create the input and output arrays in device memory for our calculation
    //
    pattern_input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(char) * pattern_length, NULL, NULL);
    search_input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(char) * search_length, NULL, NULL);
    line_lengths_input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(int) * MAX_NUMBER_OF_LINES, NULL, NULL);
    results_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned long) * results_count, NULL, NULL);
    if (!pattern_input || !search_input || !results_output)
    {
        Warning("ERROR: Failed to allocate device memory!\n");
        exit(1);
    }    
    
    // Write our data set into the input array in device memory 
    //
    err = clEnqueueWriteBuffer(commands, pattern_input, CL_TRUE, 0, sizeof(char) * pattern_length, pattern_string, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        Warning("ERROR: Failed to write to input pattern string device memory! %s\n", GetErrorString(err));
        exit(1);
    }

    err = clEnqueueWriteBuffer(commands, search_input, CL_TRUE, 0, sizeof(char) * search_length, search_string, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        Warning("ERROR: Failed to write to input search string device memory! %s\n", GetErrorString(err));
        exit(1);
    }

    err = clEnqueueWriteBuffer(commands, line_lengths_input, CL_TRUE, 0, sizeof(int) * MAX_NUMBER_OF_LINES, linesizes, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        Warning("ERROR: Failed to write to input search string device memory! %s\n", GetErrorString(err));
        exit(1);
    }
    // Set the arguments to our compute kernel
    //
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem),		&pattern_input);
    err |= clSetKernelArg(kernel, 1, sizeof(unsigned long), &pattern_length);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem),		&search_input);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned long), &search_length);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem),		&line_lengths_input);
    err |= clSetKernelArg(kernel, 5, sizeof(unsigned long), &number_of_lines_read);
    err |= clSetKernelArg(kernel, 6, sizeof(cl_mem),		&results_output);
    if (err != CL_SUCCESS)
    {
        Warning("ERROR: Failed to set kernel arguments! %s\n", GetErrorString(err));
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        Warning("ERROR: Failed to retrieve kernel work group info! %s\n", GetErrorString(err));
        exit(1);
    }

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    global = options.global ? options.global : search_length;
    local = options.local ? (options.local <= local ? options.local : local) : local;
    global = local > global ? local : global;
    iterations = options.iterations ? options.iterations : 1;
    if(!options.quiet)
    {
        Info(SEPARATOR);
        Info("Executing '%s' (global='%d, local='%d') for %d iterations...\n", 
            KernelMethodName, (int)global, (int)local, iterations);
    }
    
    // Get the current host time for the host if timing was requested
    //
    if(options.timing)
    {
        frequency = GetTimerFrequencyForHost();
        t0 = t1 = GetCurrentTimeForHost();
    }

    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        Warning("ERROR: Failed to execute kernel! %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    // Wait for the command commands to get serviced before reading back results
    //
    clFinish(commands);

    // Calculate the statistics for execution time and throughput
    //
    if(options.timing)
    {
        t1 = GetCurrentTimeForHost();
        double t = SubtractTimeInSecondsForHost(t1, t0, frequency);
        Info(SEPARATOR);
        Info("Exec Time: %.2f ms (for %d iterations)\n", 1000.0 * t / (double)(iterations),(int) iterations);
        Info(SEPARATOR);
    }
    
    // Read back the results from the device to verify the output
    //
    err = clEnqueueReadBuffer( commands, results_output, CL_TRUE, 0, sizeof(unsigned long) * results_count, results, 0, NULL, NULL );  
    if (err != CL_SUCCESS)
    {
        Warning("ERROR: Failed to read output array! %s\n", GetErrorString(err));
        exit(1);
    }
  
    int num = 0; 
    for(int h = 0 ; h < number_of_lines_read; h++)
    {
        printf("\n %d: %d", h, results[h]);
        if(results[h] == 1)
            num += 1;
    }
    // Verify our results
    //
    if(options.verify)
    {
        Info("Verifying search results...\n");
        unsigned long found = VerifyPatternCountInSearchBuffer(pattern_string, pattern_length, search_string, search_length);
        Info("Found '%lu' out of '%d' expected patterns!\n", (unsigned long)found, num);
        Info(SEPARATOR);
    }
    
    // Shutdown and cleanup
    //
    clReleaseMemObject(pattern_input);
    clReleaseMemObject(search_input);
    clReleaseMemObject(results_output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    
    free(kernel_source);
    free(pattern_string);
    free(search_string);
    
    return 0;
}
