#ifndef GENESIS_DRIVER_H
#define GENESIS_DRIVER_H

#include <sys/mman.h>

#define MAP_SIZE 4096UL
#define AXI4_ADDR_BASE  0x43C00000
#define IP0_ADDR_BASE   AXI4_ADDR_BASE | 0x0000
#define IP1_ADDR_BASE   AXI4_ADDR_BASE | 0x1000


//TODO: const char *bitstream_path = "";

// AXI4-Lite Peek-Poker
struct axi4lpp{
    int dev_mem_fd;
    uint32_t *base_ip0;
    uint32_t *base_ip1;
};

// Init
void axi4lpp_init(struct axi4lpp *pp);
void _axi4lpp_open_dev_mem(int *dev_mem_fd);
void _axi4lpp_make_map(off_t dev_base_addr, uint32_t **mapped_base_addr, int *dev_mem_fd);
//TODO: void program_bitstream();
//TODO: int get_bitstream_program_status();

// Delete
void axi4lpp_del(struct axi4lpp *pp);
void _axi4lpp_close_map(uint32_t **mapped_base_addr);
void _axi4lpp_close_dev_mem(int *dev_mem_fd);

// AXI4-Lite read and write
uint32_t _axi4lpp_axi_read(uint32_t *base_addr, uint32_t word_addr);
void _axi4lpp_axi_write(uint32_t *base_addr, uint32_t word_addr, uint32_t data);

#endif
