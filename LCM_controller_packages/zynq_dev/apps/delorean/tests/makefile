# Automatic Variables
# (https://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html)
#
#   $@ - The file name of the target of the rule.
#   $^ - The names of all the prerequisites, with spaces between them.
#   $< - The name of the first prerequisite.
#
# Rules: targets, prerequisites, and recipes
#   1. A target is out of date if 1) it does not exist or 2) it is older than
#      any of the prerequisites.
#   2. The recipe describes how to update the target, after all prerequisites
#      are met.
#   3. Prerequisites are not necessarily evaluated in the order listed.
#   4. Prerequisites don't need to have corresponding targets (e.g. they may
#      be sources)
#   5. Targets don't need to have prerequisites (e.g. 'clean' runs it recipe
#      for 'make clean' because a file 'clean' doesn't exist--see item 1 above).
#
# Functions:
#   $(patsubst pattern,replacement,$(var))
#      Replace 'pattern' with 'replacement' in variable 'var'. '%' is wildcard.

# -----------------------------------------------------------------------------
# Variables
# -----------------------------------------------------------------------------
IDIR = ..
ODIR = .
CC = gcc
CC_FLAGS = -Wall -lm -I$(IDIR) -std=c99 -D_POSIX_C_SOURCE=199309L

DESIGN = delorean
SOURCES = $(IDIR)/$(DESIGN)_driver.c
HEADERS = $(SOURCES:.c=.h)
OBJECTS = $(patsubst $(IDIR)/%.c,$(ODIR)/%.o,$(SOURCES))

MAINS = $(wildcard *.c)
MOBJECTS = $(MAINS:.c=.o)
ELFS = $(MAINS:.c=.elf)

# -----------------------------------------------------------------------------
# Targets
# -----------------------------------------------------------------------------
all: $(ELFS)

%.elf: %.o $(OBJECTS)
	$(CC) $(CC_FLAGS) -o $@ $^

# This is specifically for files in MOBJECTS
%.o: %.c $(HEADERS)
	$(CC) $(CC_FLAGS) -o $@ -c $<

# This is specifically for files in OBJECTS
$(ODIR)/%.o: $(IDIR)/%.c $(HEADERS)
	$(CC) $(CC_FLAGS) -o $@ -c $<

print:
	@echo IDIR '    ' $(IDIR)
	@echo ODIR '    ' $(ODIR)
	@echo SOURCES ' ' $(SOURCES)
	@echo HEADERS ' ' $(HEADERS)
	@echo OBJECTS ' ' $(OBJECTS)
	@echo MAINS '   ' $(MAINS)
	@echo MOBJECTS '' $(MOBJECTS)
	@echo ELFS '    ' $(ELFS)

clean:
	rm -f $(ELFS) $(MOBJECTS) $(OBJECTS)
