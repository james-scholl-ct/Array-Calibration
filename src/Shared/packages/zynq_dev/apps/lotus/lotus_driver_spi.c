#include <stdint.h>

#include "lotus_driver_spi.h"

const int LOTUS_SPI_SWITCH_MAP[204][2] = {
    {1, 31}, {0, 18}, {2, 15}, {0, 10}, {0, 22}, {0,  1}, {3, 14}, {0, 14}, // 0
    {0, 30}, {0,  2}, {0,  6}, {0, 11}, {1,  6}, {0, 24}, {2,  6}, {0, 13}, // 1
    {1, 14}, {1,  8}, {3, 23}, {0,  8}, {1, 23}, {0, 25}, {1, 22}, {0, 12}, // 3
    {1, 30}, {1,  9}, {2, 22}, {0,  9}, {2,  7}, {1, 10}, {2, 31}, {0, 26}, // 4
    {2, 14}, {1, 25}, {4,  6}, {0,  0}, {3, 15}, {1, 26}, {3, 30}, {0, 16}, // 5
    {2, 23}, {1,  0}, {2, 30}, {0, 17}, {3,  6}, {1, 16}, {4,  7}, {1,  1}, // 6
    {3,  7}, {2, 10}, {4, 22}, {1,  2}, {4, 15}, {1, 17}, {3, 31}, {1, 18}, // 7
    {3, 22}, {2,  2}, {4, 14}, {1, 24}, {4, 30}, {2,  1}, {4, 23}, {2,  9}, // 8
    {4, 31}, {2,  0}, {4, 29}, {2,  8}, {5, 15}, {2, 24}, {6,  7}, {2, 26}, // 9
    {4, 21}, {2, 18}, {5, 13}, {2, 25}, {6, 15}, {3,  9}, {5, 14}, {2, 16}, //10
    {5, 23}, {3,  2}, {5, 31}, {3, 10}, {5, 22}, {2, 17}, {6, 14}, {3,  0}, //11
    {5, 30}, {3,  8}, {5,  7}, {3,  1}, {5, 29}, {3, 17}, {5,  6}, {3, 26}, //12
    {5,  5}, {3, 24}, {6, 13}, {3, 16}, {5,  3}, {1, 15}, {5, 21}, {4, 10}, //13
    {5, 11}, {4,  0}, {6, 12}, {4,  9}, {5,  4}, {3, 18}, {6, 11}, {4, 26}, //14
    {5, 28}, {4, 17}, {4,  3}, {4, 16}, {5, 10}, {3, 25}, {3, 19}, {5,  0}, //15
    {5, 27}, {4,  1}, {3, 20}, {5, 26}, {5,  9}, {6,  0}, {4,  4}, {5, 16}, //16
    {5, 20}, {4,  8}, {4, 25}, {6, 10}, {4,  5}, {4, 18}, {3,  4}, {5, 25}, //17
    {5,  8}, {4,  2}, {4, 24}, {6,  9}, {4, 27}, {0,  5}, {3, 21}, {5, 24}, //18
    {4, 20}, {5,  2}, {2, 19}, {6,  3}, {5, 12}, {5, 18}, {5, 19}, {6,  8}, //19
    {4, 28}, {5,  1}, {3,  5}, {6,  4}, {4, 19}, {0,  4}, {1, 27}, {6,  2}, //20
    {4, 12}, {5, 17}, {4, 11}, {6,  5}, {4, 13}, {0, 21}, {3, 13}, {6,  6}, //21
    {3, 28}, {1,  5}, {3, 27}, {6,  1}, {3, 29}, {0,  3}, {0, 27}, {1,  7}, //22
    {3, 11}, {0, 20}, {2, 11}, {1, 21}, {3,  3}, {1,  4}, {3, 12}, {2,  5}, //23
    {2, 21}, {0, 19}, {1, 11}, {1, 20}, {2, 20}, {0, 29}, {0, 28}, {2,  4}, //24
    {2, 12}, {1,  3}, {2, 27}, {1, 19}, {2, 28}, {1, 29}, {1, 28}, {2,  3}, //25
    {2, 13}, {1, 13}, {1, 12}, {2, 29}                                      //26
};


// ----------------------------------------------------------------------------
// Generic read and write
// ----------------------------------------------------------------------------
uint32_t axi4lpp_spi_read(struct axi4lpp *pp, uint32_t word_addr)
{
    return _axi4lpp_axi_read(pp->base_spi, word_addr);
}

void axi4lpp_spi_write(struct axi4lpp *pp, uint32_t word_addr, uint32_t data)
{
    _axi4lpp_axi_write(pp->base_spi, word_addr, data);
}

// ----------------------------------------------------------------------------
// Config
// ----------------------------------------------------------------------------
void axi4lpp_spi_config_clkdivs(struct axi4lpp *pp, uint32_t data)
{
    axi4lpp_spi_write(pp, LOTUS_SPI_CLKDIV_WORD, data);
}

void axi4lpp_spi_set_config(struct axi4lpp *pp, uint32_t data)
{
    axi4lpp_spi_write(pp, LOTUS_SPI_CONFIG_WORD, data);
}

uint32_t axi4lpp_spi_get_config(struct axi4lpp *pp)
{
    return axi4lpp_spi_read(pp, LOTUS_SPI_CONFIG_WORD);
}

// ----------------------------------------------------------------------------
// Control
// ----------------------------------------------------------------------------
void axi4lpp_spi_send_cmd(struct axi4lpp *pp,
                          enum axi4lpp_spi_cmd_e cmd,
                          enum axi4lpp_spi_slave_e slave,
                          uint32_t payload)
{
    uint32_t data = (uint32_t)cmd << 28 | (uint32_t)slave << 24 | payload;
    axi4lpp_spi_write(pp, LOTUS_SPI_CMD_WORD, data);
}

uint32_t axi4lpp_spi_read_rsp(struct axi4lpp *pp)
{
    return axi4lpp_spi_read(pp, LOTUS_SPI_RSP_WORD);
}

uint32_t axi4lpp_spi_get_rsp_is_valid(struct axi4lpp *pp, uint32_t rsp)
{
    return (uint32_t)(rsp >> 31 == 1);
}

uint32_t axi4lpp_spi_get_rsp_fifo_count(struct axi4lpp *pp, uint32_t rsp)
{
    return (uint32_t)(rsp >> 19 & 0x3f);
}

enum axi4lpp_spi_slave_e axi4lpp_spi_get_rsp_slave_idx(struct axi4lpp *pp,
                                                       uint32_t rsp)
{
    return (enum axi4lpp_spi_slave_e)(rsp >> 16 & 0x7);
}

uint32_t axi4lpp_spi_get_rsp_payload(struct axi4lpp *pp, uint32_t rsp)
{
    return (uint32_t)(rsp & 0xffff);
}

// ----------------------------------------------------------------------------
// Status
// ----------------------------------------------------------------------------
uint32_t axi4lpp_spi_read_status(struct axi4lpp *pp)
{
    return axi4lpp_spi_read(pp, LOTUS_SPI_STATUS_WORD);
}

uint32_t axi4lpp_spi_get_cmd_fifo_is_full(struct axi4lpp *pp)
{
    return axi4lpp_spi_read(pp, LOTUS_SPI_STATUS_WORD) >> 4 & 1;
}

uint32_t axi4lpp_spi_get_cmd_fifo_count(struct axi4lpp *pp)
{
    return axi4lpp_spi_read(pp, LOTUS_SPI_STATUS_WORD) & 0xf;
}

uint32_t axi4lpp_spi_get_cmd_fifo_headroom(struct axi4lpp *pp)
{
    return 15 - axi4lpp_spi_get_cmd_fifo_count(pp);
}

// ----------------------------------------------------------------------------
// Data
// ----------------------------------------------------------------------------
void axi4lpp_spi_write_jumbo_frame_data(struct axi4lpp *pp,
                                        uint32_t spi0_words[7],
                                        uint32_t spi1_words[7])
{
    for (int i = 0; i < 7; i = i + 1)
    {
        axi4lpp_spi_write(pp, LOTUS_SPI_TABLE0_START_WORD + i, spi0_words[i]);
        axi4lpp_spi_write(pp, LOTUS_SPI_TABLE1_START_WORD + i, spi1_words[i]);
    }
}
