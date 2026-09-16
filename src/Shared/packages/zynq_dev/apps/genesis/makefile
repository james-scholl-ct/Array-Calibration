# Help: Automatic Variables
# (https://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html)
#
# $@ - The file name of the target of the rule.
# $^ - The names of all the prerequisites, with spaces between them.
# $< - The name of the first prerequisite.

# -----------------------------------------------------------------------------
# Variables
# -----------------------------------------------------------------------------
CC = gcc
CC_FLAGS = -Wall -lm -std=c99 -D_POSIX_C_SOURCE=199309L

DESIGN = genesis
PROG = $(DESIGN)_app
SOURCES = $(DESIGN)_driver.c \
          $(DESIGN)_driver_ip0.c \
          $(DESIGN)_driver_ip1.c
MAINS = $(PROG).c
OBJECTS = $(SOURCES:.c=.o)
MOBJECTS = $(MAINS:.c=.o)
ELFS = $(MAINS:.c=.elf)

# -----------------------------------------------------------------------------
# Targets
# -----------------------------------------------------------------------------
all: $(ELFS)

$(PROG).elf: $(PROG).o $(OBJECTS)
	$(CC) $(CC_FLAGS) -o $@ $^
	mkdir -p /root/$(DESIGN)
	cp $(PROG).elf /root/$(DESIGN)/$(PROG).elf

%.elf: %.o $(OBJECTS)
	$(CC) $(CC_FLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CC_FLAGS) -c $<

clean:
	rm -f $(ELFS) $(MOBJECTS) $(OBJECTS)
