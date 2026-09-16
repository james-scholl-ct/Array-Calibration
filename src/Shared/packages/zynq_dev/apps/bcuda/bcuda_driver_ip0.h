#ifndef BCUDA_DRIVER_IP0_H
#define BCUDA_DRIVER_IP0_H

#include "bcuda_driver.h"

// Generic read and write
uint32_t axi4lpp_ip0_read(struct axi4lpp *pp, uint32_t word_addr);
void axi4lpp_ip0_write(struct axi4lpp *pp, uint32_t word_addr, uint32_t data);

#endif
