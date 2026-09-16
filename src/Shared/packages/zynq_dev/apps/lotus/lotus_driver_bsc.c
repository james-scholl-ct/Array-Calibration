#include <stdint.h>

#include "lotus_driver_bsc.h"


// ----------------------------------------------------------------------------
// Generic read and write
// ----------------------------------------------------------------------------
uint32_t axi4lpp_bsc_read(struct axi4lpp *pp, uint32_t word_addr)
{
    return _axi4lpp_axi_read(pp->base_bsc, word_addr);
}

void axi4lpp_bsc_write(struct axi4lpp *pp, uint32_t word_addr, uint32_t data)
{
    _axi4lpp_axi_write(pp->base_bsc, word_addr, data);
}

// ----------------------------------------------------------------------------
// Config
// ----------------------------------------------------------------------------
void axi4lpp_bsc_set_dwell(struct axi4lpp *pp, uint32_t dwell_count)
{
    axi4lpp_bsc_write(pp, LOTUS_BSC_DWELL_WORD, dwell_count);
}

uint32_t axi4lpp_bsc_dwell_cnt_from_us(struct axi4lpp *pp, float time_us)
{
    // truncates fractional part
    return (uint32_t)(pp->bsc_clk_freq_MHz * time_us - 4);
}

float axi4lpp_bsc_dwell_cnt_to_us(struct axi4lpp *pp, uint32_t dwell_count)
{
    return (dwell_count + 4) / pp->bsc_clk_freq_MHz;
}

// ----------------------------------------------------------------------------
// Control
// ----------------------------------------------------------------------------
void axi4lpp_bsc_reset(struct axi4lpp *pp)
{
    axi4lpp_bsc_write(pp, LOTUS_BSC_CONTROL_WORD, 1<<0 | pp->pol_ovr << 4);
}

void axi4lpp_bsc_init(struct axi4lpp *pp)
{
    axi4lpp_bsc_write(pp, LOTUS_BSC_CONTROL_WORD, 1<<1 | pp->pol_ovr << 4);
}

void axi4lpp_bsc_apply_table(struct axi4lpp *pp)
{
    uint32_t data = 1<<2 | 1<<3 | pp->pol_ovr << 4;
    axi4lpp_bsc_write(pp, LOTUS_BSC_CONTROL_WORD, data);
}

void axi4lpp_bsc_stop(struct axi4lpp *pp)
{
    axi4lpp_bsc_write(pp, LOTUS_BSC_CONTROL_WORD, 0 | pp->pol_ovr << 4);
}

void axi4lpp_bsc_transfer_table(struct axi4lpp *pp, uint32_t table_index)
{
    axi4lpp_bsc_write(pp, LOTUS_BSC_TXFER_WORD, table_index);
}

// ----------------------------------------------------------------------------
// Status
// ----------------------------------------------------------------------------
uint32_t axi4lpp_bsc_read_status(struct axi4lpp *pp)
{
    return axi4lpp_bsc_read(pp, LOTUS_BSC_STATUS_WORD);
}

uint32_t axi4lpp_bsc_driver_is_done(struct axi4lpp *pp)
{
    return axi4lpp_bsc_read(pp, LOTUS_BSC_STATUS_WORD) >> 1 & 1;
}

uint32_t axi4lpp_bsc_table_transfer_is_busy(struct axi4lpp *pp)
{
    return axi4lpp_bsc_read(pp, LOTUS_BSC_STATUS_WORD) & 1;
}

// ----------------------------------------------------------------------------
// Data
// ----------------------------------------------------------------------------
void axi4lpp_bsc_set_table(struct axi4lpp *pp, uint8_t table[204])
{
    for (int i = 0; i < 204/4; i = i + 1)
    {
        uint32_t data = table[4*i+3] << 8*3 |
                        table[4*i+2] << 8*2 |
                        table[4*i+1] << 8*1 |
                        table[4*i+0] << 8*0;
        axi4lpp_bsc_write(pp, LOTUS_BSC_TABLE_START_WORD + i, data);
    }
}

void axi4lpp_bsc_get_table(struct axi4lpp *pp, uint8_t table[204])
{
    for (int i = 0; i < 204/4; i = i + 1)
    {
        uint32_t data = axi4lpp_bsc_read(pp, LOTUS_BSC_TABLE_START_WORD + i);
        table[4*i+3] = data >> 8*3 & 0xff;
        table[4*i+2] = data >> 8*2 & 0xff;
        table[4*i+1] = data >> 8*1 & 0xff;
        table[4*i+0] = data >> 8*0 & 0xff;
    }
}
