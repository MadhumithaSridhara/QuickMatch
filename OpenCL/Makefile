TARGET = pm

OS_NAME 		:= $(shell uname -s)

ifeq ($(OS_NAME),Darwin)
	DEFINES		+= -DDARWIN -D__APPLE__ -D__apple__
	INCLUDE		+= -I. -I/usr/include -I/System/Library/Frameworks/OpenCL.framework/Headers
	LIBRARIES 	+= -framework OpenCL -framework AppKit -framework Foundation
endif

ifeq ($(OS_NAME),Linux)
	DEFINES		+= -DLINUX -D__LINUX__ -D__linux__
	INCLUDE		+= -I. -I/usr/include -I/usr/local/include
	LIBPATH 	+= -L. -L/usr/lib
	LIBRARIES 	+= -lOpenCL
endif

OBJDIR 			= .
HEADERS 		= $(shell ls *.h) 
SOURCES 		= $(shell ls *.cpp *.c)
OBJECTS 		= $(patsubst %.cpp,$(OBJDIR)/%.o,$(SOURCES))
OBJECTS 		:= $(patsubst %.c,$(OBJDIR)/%.o,$(OBJECTS))

RM				= rm
CC 				= g++
WARN 			= -Wall
CFLAGS 			= -c -g -O3 $(INCLUDE) $(WARN)
	
$(OBJDIR)/%.o: %.cpp
	@echo "Building '$@'"
	$(CC) $(CFLAGS) -o $@ -c $<
	
$(OBJDIR)/%.o: %.c
	@echo "Building '$@'"
	$(CC) $(CFLAGS) -o $@ -c $<
	
$(TARGET): $(OBJECTS)
	@echo "Linking '$@'"
	$(CC) $(OBJECTS) -o $@ $(LIBPATH) $(LIBRARIES)
			
clean:
	@echo "Cleaning '$@'"
	$(RM) -f $(TARGET) $(OBJECTS)
		
.DEFAULT:
	@echo The target \"$@\" does not exist in this Makefile.
