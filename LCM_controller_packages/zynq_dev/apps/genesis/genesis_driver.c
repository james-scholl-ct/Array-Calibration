#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <math.h>

#include "genesis_driver.h"


// ----------------------------------------------------------------------------
// Init
// ----------------------------------------------------------------------------
void axi4lpp_init(struct axi4lpp *pp)
{
    _axi4lpp_open_dev_mem(&pp->dev_mem_fd);
    _axi4lpp_make_map(IP0_ADDR_BASE, &pp->base_ip0, &pp->dev_mem_fd);
    _axi4lpp_make_map(IP1_ADDR_BASE, &pp->base_ip1, &pp->dev_mem_fd);
}

void _axi4lpp_open_dev_mem(int *dev_mem_fd)
{
    // need root access to open this file
    *dev_mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (*dev_mem_fd == -1)
    {
        printf("ERROR: can't open /dev/mem.\n");
        exit(1);
    }
}

void _axi4lpp_make_map(off_t dev_base_addr, uint32_t **mapped_base_addr, int *dev_mem_fd)
{
    *mapped_base_addr = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED,
                             *dev_mem_fd, dev_base_addr);
    if (*mapped_base_addr == (uint32_t *) -1)
    {
        printf("ERROR: can't map mem to 0x%x.\n", (unsigned int)dev_base_addr);
        exit(1);
    }
}

// ----------------------------------------------------------------------------
// Delete
// ----------------------------------------------------------------------------
void axi4lpp_del(struct axi4lpp *pp)
{
    _axi4lpp_close_map(&pp->base_ip1);
    _axi4lpp_close_map(&pp->base_ip0);
    _axi4lpp_close_dev_mem(&pp->dev_mem_fd);
}

void _axi4lpp_close_map(uint32_t **mapped_base_addr)
{
    int status = munmap(*mapped_base_addr, MAP_SIZE);
    if (status == -1)
    {
        printf("ERROR: couldn't unmap memory from user space.\n");
    }
}

void _axi4lpp_close_dev_mem(int *dev_mem_fd)
{
    close(*dev_mem_fd);
}

// ----------------------------------------------------------------------------
// AXI4-Lite read and write
// ----------------------------------------------------------------------------
uint32_t _axi4lpp_axi_read(uint32_t *base_addr, uint32_t word_addr)
{
    volatile uint32_t *addr = base_addr + word_addr;
    return *addr;
}

void _axi4lpp_axi_write(uint32_t *base_addr, uint32_t word_addr, uint32_t data)
{
    volatile uint32_t *addr = base_addr + word_addr;
    *addr = data;
}
