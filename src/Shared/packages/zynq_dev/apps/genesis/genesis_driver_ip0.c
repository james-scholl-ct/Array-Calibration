#include <stdint.h>

#include "genesis_driver_ip0.h"


// ----------------------------------------------------------------------------
// Generic read and write
// ----------------------------------------------------------------------------
uint32_t axi4lpp_ip0_read(struct axi4lpp *pp, uint32_t word_addr)
{
    return _axi4lpp_axi_read(pp->base_ip0, word_addr);
}

void axi4lpp_ip0_write(struct axi4lpp *pp, uint32_t word_addr, uint32_t data)
{
    _axi4lpp_axi_write(pp->base_ip0, word_addr, data);
}
