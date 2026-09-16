#ifndef GENESIS_DRIVER_IP1_H
#define GENESIS_DRIVER_IP1_H

#include "genesis_driver.h"

// Generic read and write
uint32_t axi4lpp_ip1_read(struct axi4lpp *pp, uint32_t word_addr);
void axi4lpp_ip1_write(struct axi4lpp *pp, uint32_t word_addr, uint32_t data);

#endif
