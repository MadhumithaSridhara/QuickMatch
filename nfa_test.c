#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_CHAR 1024

int characters[MAX_CHAR];

void build_characters(char *regex) {
	int i;
	int length = strlen(regex);
	for (i=0; i<length; i++) {
		characters[i] = regex[i];
	}
	characters[i] = '\0';
}

int match(char *line) {
	int i, counter;
	int length = strlen(line);
	for (i=0, counter=0; i<length; i++, counter++) {
		if (characters[counter] == '\0') {
			return 1;
		}
		if (characters[counter] != line[i]) {
			counter = 0;
		}
	}
	return 0;
}


int main(int argc, char **argv) {
	int i;

	if(argc < 3) {
		fprintf(stderr, "usage: nfa regexp string...\n");
		return 1;
	}

	build_characters(argv[1]);

	FILE *fp;
    char *line = NULL;
    size_t len = 0;
    ssize_t read;

    // File input
   	fp = fopen(argv[2], "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

	while ((read = getline(&line, &len, fp)) != -1) {
		line[read-1] = '\0';
	    if(match(line))
			printf("%s\n", line);
	}

   	free(line);
		
	return 0;
}