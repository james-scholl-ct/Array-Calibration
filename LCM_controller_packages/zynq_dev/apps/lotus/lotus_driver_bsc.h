#ifndef LOTUS_DRIVER_BSC_H
#define LOTUS_DRIVER_BSC_H

#include "lotus_driver.h"

#define LOTUS_BSC_TABLE_START_WORD 0
#define LOTUS_BSC_TXFER_WORD 60
#define LOTUS_BSC_DWELL_WORD 61
#define LOTUS_BSC_CONTROL_WORD 62
#define LOTUS_BSC_STATUS_WORD 63

// Generic read and write
uint32_t axi4lpp_bsc_read(struct axi4lpp *pp, uint32_t word_addr);
void axi4lpp_bsc_write(struct axi4lpp *pp, uint32_t word_addr, uint32_t data);

// Config
void axi4lpp_bsc_set_dwell(struct axi4lpp *pp, uint32_t dwell_count);
uint32_t axi4lpp_bsc_dwell_cnt_from_us(struct axi4lpp *pp, float time_us);
float axi4lpp_bsc_dwell_cnt_to_us(struct axi4lpp *pp, uint32_t dwell_count);

// Control
void axi4lpp_bsc_reset(struct axi4lpp *pp);
void axi4lpp_bsc_init(struct axi4lpp *pp);
void axi4lpp_bsc_apply_table(struct axi4lpp *pp);
void axi4lpp_bsc_stop(struct axi4lpp *pp);
void axi4lpp_bsc_transfer_table(struct axi4lpp *pp, uint32_t table_index);

// Status
uint32_t axi4lpp_bsc_read_status(struct axi4lpp *pp);
uint32_t axi4lpp_bsc_driver_is_done(struct axi4lpp *pp);
uint32_t axi4lpp_bsc_table_transfer_is_busy(struct axi4lpp *pp);

// Data
void axi4lpp_bsc_set_table(struct axi4lpp *pp, uint8_t table[204]);
void axi4lpp_bsc_get_table(struct axi4lpp *pp, uint8_t table[204]);

#endif
