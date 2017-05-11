
int matchme(char* line, int *characters, int linelength);
void __kernel PatternMatcher(                                                 
   __global char* pattern_string, const unsigned long pattern_length,
   __global char* buffer_string,  const unsigned long buffer_length,                                             
   __global int* line_offsets,  const unsigned int number_of_lines,                                             
   __global unsigned long* results)                                                                                   
{                                                                      
     int i;
     uint global_addr = get_global_id(0);
     char characters[80]; 
     if(global_addr > number_of_lines)
        return;
	for(i=0; i<pattern_length; i++) 
    {
		characters[i] = pattern_string[i];
	}
	characters[i] = '\0';
    
    int offset = line_offsets[global_addr];

    results[global_addr] = 0;

    int counter;
    counter = 0;
	for(int k=0; k<buffer_length;k++) 
    {
		if (characters[counter] == '\0') {
            results[global_addr] = 1;
            return;
		}
		if (characters[counter] != buffer_string[k+offset]) {
            if(global_addr==0)
                printf("%c", buffer_string[k]);
			counter = 0;
		}
        counter++;
	}
}                                                                  
int matchme(char* line, int *characters, int linelength) 
{
    int counter;
    counter = 0;
	for(int k=0; k<linelength;k++) 
    {
		if (characters[counter] == '\0') {
			return 1;
		}
		if (characters[counter] != line[k]) {
			counter = 0;
		}
        counter++;
	}
	return 0;
}
