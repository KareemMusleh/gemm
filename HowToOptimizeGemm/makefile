OLD  := MMult0
NEW  := MMult0
#
# sample makefile
#

CC         := gcc
LINKER     := $(CC)
CFLAGS     := -O2 -Wall -msse3
LDFLAGS    := -lm
OBJ_DIR  := obj

UTIL       := $(OBJ_DIR)/copy_matrix.o \
              $(OBJ_DIR)/compare_matrices.o \
              $(OBJ_DIR)/random_matrix.o \
              $(OBJ_DIR)/dclock.o \
              $(OBJ_DIR)/REF_MMult.o \
              $(OBJ_DIR)/print_matrix.o

TEST_OBJS  := $(OBJ_DIR)/test_MMult.o $(OBJ_DIR)/$(NEW).o 

$(OBJ_DIR)/%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

all: 
	make clean;
	make test_MMult.x

test_MMult.x: $(TEST_OBJS) $(UTIL) parameters.h
	$(LINKER) $(TEST_OBJS) $(UTIL) $(LDFLAGS) \
        $(BLAS_LIB) -o $(TEST_BIN) $@ 

run:	
	make all
	export OMP_NUM_THREADS=1
	export GOTO_NUM_THREADS=1
	echo "version = '$(NEW)';" > output_$(NEW).m
	./test_MMult.x >> output_$(NEW).m
	cp output_$(OLD).m output_old.m
	cp output_$(NEW).m output_new.m

clean:
	rm -f $(OBJ_DIR)/*.o *~ core *.x

cleanall:
	rm -f $(OBJ_DIR)/*.o *~ core *.x output*.m *.eps *.png
