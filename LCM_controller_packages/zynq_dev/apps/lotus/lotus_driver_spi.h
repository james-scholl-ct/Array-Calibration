#ifndef LOTUS_DRIVER_SPI_H
#define LOTUS_DRIVER_SPI_H

#include "lotus_driver.h"

#define LOTUS_SPI_TABLE0_START_WORD 0
#define LOTUS_SPI_TABLE1_START_WORD 8
#define LOTUS_SPI_CLKDIV_WORD 24
#define LOTUS_SPI_CONFIG_WORD 25
#define LOTUS_SPI_CMD_WORD 26
#define LOTUS_SPI_RSP_WORD 30
#define LOTUS_SPI_STATUS_WORD 31

#define LOTUS_SPI_CONFIG_POS_FILL_CMD_FIFO 0
#define LOTUS_SPI_CONFIG_MSK_FILL_CMD_FIFO 1
#define LOTUS_SPI_CONFIG_POS_ADC0_CHSEL 1
#define LOTUS_SPI_CONFIG_MSK_ADC0_CHSEL 1
#define LOTUS_SPI_CONFIG_POS_DAISY_EN 2
#define LOTUS_SPI_CONFIG_MSK_DAISY_EN 1

enum axi4lpp_spi_cmd_e {STANDARD, JUMBO, STREAM};
enum axi4lpp_spi_slave_e {ADC0, TEMP, ADC1, DAISY, CAL};
// Maps rail index (0-203) to a pair (axi word, bit in word)
const int LOTUS_SPI_SWITCH_MAP[204][2];

// Generic read and write
uint32_t axi4lpp_spi_read(struct axi4lpp *pp, uint32_t word_addr);
void axi4lpp_spi_write(struct axi4lpp *pp, uint32_t word_addr, uint32_t data);

// Config
void axi4lpp_spi_config_clkdivs(struct axi4lpp *pp, uint32_t data);
void axi4lpp_spi_set_config(struct axi4lpp *pp, uint32_t data);
uint32_t axi4lpp_spi_get_config(struct axi4lpp *pp);

// Control
void axi4lpp_spi_send_cmd(struct axi4lpp *pp,
                          enum axi4lpp_spi_cmd_e cmd,
                          enum axi4lpp_spi_slave_e slave,
                          uint32_t payload);
uint32_t axi4lpp_spi_read_rsp(struct axi4lpp *pp);
uint32_t axi4lpp_spi_get_rsp_is_valid(struct axi4lpp *pp, uint32_t rsp);
uint32_t axi4lpp_spi_get_rsp_fifo_count(struct axi4lpp *pp, uint32_t rsp);
enum axi4lpp_spi_slave_e axi4lpp_spi_get_rsp_slave_idx(struct axi4lpp *pp,
                                                       uint32_t rsp);
uint32_t axi4lpp_spi_get_rsp_payload(struct axi4lpp *pp, uint32_t rsp);

// Status
uint32_t axi4lpp_spi_read_status(struct axi4lpp *pp);
uint32_t axi4lpp_spi_get_cmd_fifo_is_full(struct axi4lpp *pp);
uint32_t axi4lpp_spi_get_cmd_fifo_count(struct axi4lpp *pp);
uint32_t axi4lpp_spi_get_cmd_fifo_headroom(struct axi4lpp *pp);

// Data
void axi4lpp_spi_write_jumbo_frame_data(struct axi4lpp *pp,
                                        uint32_t spi0_words[7],
                                        uint32_t spi1_words[7]);

#endif
