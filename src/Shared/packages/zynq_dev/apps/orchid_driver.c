#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <math.h>

#include "orchid_driver.h"


int setup_map(uint32_t **mapped_base_addr, int *dev_mem_fd)
{
    off_t dev_base_addr = ORCHID_AXI4_ADDR_BASE;

    // need root access to open this file
    *dev_mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (*dev_mem_fd == -1)
    {
        printf("ERROR: can't open /dev/mem.\n");
        exit(1);
    }

    *mapped_base_addr = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED,
                            *dev_mem_fd, dev_base_addr);
    if (*mapped_base_addr == (uint32_t *) -1)
    {
        printf("ERROR: can't map the memory to user space.\n");
        exit(1);
    }

    return 0;
}


void close_map(uint32_t **mapped_base_addr, int *dev_mem_fd)
{
    int status = munmap(*mapped_base_addr, MAP_SIZE);
    if (status == -1)
    {
        printf("ERROR: couldn't unmap memory from user space.\n");
    }

    close(*dev_mem_fd);
}


void axi_write(uint32_t *base_addr, uint32_t word_addr, uint32_t data)
{
    volatile uint32_t *addr = base_addr + word_addr;
    *addr = data;
}


uint32_t axi_read(uint32_t *base_addr, uint32_t word_addr)
{
    volatile uint32_t *addr = base_addr + word_addr;
    return *addr;
}


int parse_coeffs_file(const char *filename, uint32_t *array)
{
    FILE *fptr = fopen(filename, "r");
    if (fptr == NULL)
    {
        printf("Cannot open input file %s\n", filename);
        exit(1);
    }

    int value;
    int i = 0;
    while(fscanf(fptr, "%d", &value) > 0)
    {
        array[i] = (uint32_t) value;
        i++;
    }

    fclose(fptr);
    return 0;
}


uint32_t set_dwell_time_us(uint32_t *base, float dwell_time_us)
{
    // 60MHz clock
    uint32_t dwell_cycles = round(dwell_time_us * 60.);
    axi_write(base, ORCHID_AXI4_DWELL_REG, dwell_cycles);
    return axi_read(base, ORCHID_AXI4_DWELL_REG);
}


void send_reset(uint32_t *base)
{
    uint32_t data = 1 << ORCHID_AXI4_CONTROL_IDX_RESET;
    axi_write(base, ORCHID_AXI4_CONTROL_REG, data);
}


void send_start(uint32_t *base, uint32_t mode)
{
    if (mode < 0 || mode > 3)
    {
        printf("ERROR: mode must be 0, 1, 2, or 3.\n");
    }
    else
    {
        uint32_t data = 1 << ORCHID_AXI4_CONTROL_IDX_START;
        data |= mode << ORCHID_AXI4_CONTROL_IDX_MODE;
        axi_write(base, ORCHID_AXI4_CONTROL_REG, data);
    }
}


void send_stop(uint32_t *base)
{
    uint32_t data = 0;
    axi_write(base, ORCHID_AXI4_CONTROL_REG, data);
}
