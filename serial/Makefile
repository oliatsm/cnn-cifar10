CC = nvc
CFLAGS = -Minfo=all -c11 -Wall -Wextra -march=native
# CC = gcc
# CFLAGS = -std=c11 -Wall -Wextra -march=native
LDFLAGS = -lm

PROFILE_FLAGS = -pg
DEBUG_FLAGS = -g

SRC = main_serial.c layers_serial.c malloc2D.c timer.c

SRC = main.c layers.c malloc2D.c timer.c

# Object files per build type
OBJ = $(SRC:.c=.o)
OBJ_PROFILE = $(SRC:.c=.profile.o)
OBJ_DEBUG = $(SRC:.c=.debug.o)

EXECUTABLE = cnn-cifar10
EXECUTABLE_PROFILE = cnn-cifar10-profile
EXECUTABLE_DEBUG = cnn-cifar10-debug

all: $(EXECUTABLE)

profile: $(EXECUTABLE_PROFILE)

debug: $(EXECUTABLE_DEBUG)

# Default build
$(EXECUTABLE): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Profiling build
$(EXECUTABLE_PROFILE): $(OBJ_PROFILE)
	$(CC) $(CFLAGS) $(PROFILE_FLAGS) -o $@ $^ $(LDFLAGS)

# Debugging build
$(EXECUTABLE_DEBUG): $(OBJ_DEBUG)
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) -o $@ $^ $(LDFLAGS)

# Compile normal object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compile profiling object files
%.profile.o: %.c
	$(CC) $(CFLAGS) $(PROFILE_FLAGS) -c $< -o $@

# Compile debugging object files
%.debug.o: %.c
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(EXECUTABLE) $(EXECUTABLE_PROFILE) $(EXECUTABLE_DEBUG) \
	      $(OBJ) $(OBJ_PROFILE) $(OBJ_DEBUG)
