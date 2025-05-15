CC = nvc
CFLAGS = -acc -Minfo=all -c11 -Wall -Wextra -march=native
PROFILE_FLAGS = -pg
DEBUG_FLAGS = -g

SRC = main.c layers.c malloc2D.c timer.c
OBJ = $(SRC:.c=.o)
EXECUTABLE = cnn-cifar10
EXECUTABLE_PROFILE = cnn-cifar10-profile
EXECUTABLE_DEBUG = cnn-cifar10-debug

all: $(EXECUTABLE)

profile: CFLAGS += $(PROFILE_FLAGS)
profile: $(EXECUTABLE_PROFILE)

debug: CFLAGS += $(DEBUG_FLAGS)
debug: $(EXECUTABLE_DEBUG)

$(EXECUTABLE): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^

$(EXECUTABLE_PROFILE): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^

$(EXECUTABLE_DEBUG): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(EXECUTABLE) $(EXECUTABLE_PROFILE) $(EXECUTABLE_DEBUG) $(OBJ)
