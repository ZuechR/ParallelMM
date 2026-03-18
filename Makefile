CC      := gcc
MPICC   := mpicc
CFLAGS  := -O3 -Wall -Wextra -std=c17
OMPFLAG := -fopenmp

BUILD_DIR := build

SEQ_SRC := sequentialMM.c
PAR_SRC := parallelMM.c

SEQ_BIN := $(BUILD_DIR)/sequentialMM
PAR_BIN := $(BUILD_DIR)/parallelMM

.PHONY: all clean

all: $(SEQ_BIN) $(PAR_BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(SEQ_BIN): $(SEQ_SRC) utils.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< -o $@

$(PAR_BIN): $(PAR_SRC) utils.h | $(BUILD_DIR)
	$(MPICC) $(CFLAGS) $(OMPFLAG) $< -o $@

clean:
	rm -rf $(BUILD_DIR)