#include <stdint.h>

#include "bcuda_driver_ip1.h"


// ----------------------------------------------------------------------------
// Generic read and write
// ----------------------------------------------------------------------------
uint32_t axi4lpp_ip1_read(struct axi4lpp *pp, uint32_t word_addr)
{
    return _axi4lpp_axi_read(pp->base_ip1, word_addr);
}

void axi4lpp_ip1_write(struct axi4lpp *pp, uint32_t word_addr, uint32_t data)
{
    _axi4lpp_axi_write(pp->base_ip1, word_addr, data);
}
