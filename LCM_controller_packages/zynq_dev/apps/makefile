# Declaration of variables
CC = gcc
CC_FLAGS = -Wall -lm

# File names
PROG = orchid_app
SOURCES = orchid_app.c orchid_driver.c
OBJECTS = $(SOURCES:.c=.o)

# Main target
$(PROG).elf: $(OBJECTS)
	$(CC) $(CC_FLAGS) -o $@ $(OBJECTS)
	mkdir -p /root/orchid
	cp $(PROG).elf /root/orchid/$(PROG).elf

# To obtain object files
%.o: %.c
	$(CC) $(CC_FLAGS) -c $<

# To remove generated files
clean:
	rm -f $(PROG).elf $(OBJECTS)
