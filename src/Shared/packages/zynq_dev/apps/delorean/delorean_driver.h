#ifndef DELOREAN_DRIVER_H
#define DELOREAN_DRIVER_H

#include <sys/mman.h>

#define SUCCESS 0
#define FAIL -1

// Base addresses
#define HP0_DDR_BASEADDR    0x10000000
#define BRAM_CTRL_BASEADDR  0x80000000
#define CDMA_BASEADDR       0x7e200000
#define LCM_BASEADDR        0x43c00000
#define SPI_BASEADDR        0x43c10000

// Map and buffer sizes
#define HP0_DDR_MAP_SIZE    0x200000UL  // 2MiB = 1MiB for each RX and TX
#define CDMA_MAP_SIZE       4096UL
#define LCM_MAP_SIZE        4096UL
#define SPI_MAP_SIZE        4096UL

#define DMA_BTT_MAX         2048        // 1024 bytes for each RX and TX
#define LCM_MAX_TABLES      1024        // HP0_DDR_MAP_SIZE / DMA_BTT_MAX
#define DATAPATH_BYTESIZE   4

//  - CDMA registers
#define CDMA_CDMACR_WOFFSET 0
#define CDMA_CDMASR_WOFFSET 1
#define CDMA_SA_WOFFSET     6
#define CDMA_DA_WOFFSET     8
#define CDMA_BTT_WOFFSET    10

//      - CDMA fields
#define CDMA_CDMACR_RESET_MASK          0x00000004

#define CDMA_CDMASR_IDLE_MASK           0x00000002
#define CDMA_CDMASR_ERR_INTERNAL_MASK   0x00000010
#define CDMA_CDMASR_ERR_SLAVE_MASK      0x00000020
#define CDMA_CDMASR_ERR_DECODE_MASK     0x00000040
#define CDMA_CDMASR_ERR_ALL_MASK        0x00000070

//  - LCM registers/fields
#define LCM_LCD_EN_WOFFSET                  0
#define LCM_LCD_EN_POS                      0
#define LCM_LCD_EN_MASK                     0x00000001
#define LCM_TCON_RESET_WOFFSET              1
#define LCM_TCON_RESET_POS                  0
#define LCM_TCON_RESET_MASK                 0x00000001
#define LCM_TCON_ENABLE_WOFFSET             1
#define LCM_TCON_ENABLE_POS                 1
#define LCM_TCON_ENABLE_MASK                0x00000002
#define LCM_APPLY0_WOFFSET                  2
#define LCM_APPLY0_POS                      0
#define LCM_APPLY0_MASK                     0x00000001
#define LCM_APPLY1_WOFFSET                  3
#define LCM_APPLY1_POS                      0
#define LCM_APPLY1_MASK                     0x00000001
#define LCM_TP1_PERIOD_WOFFSET              4
#define LCM_TP1_PERIOD_POS                  0
#define LCM_TP1_PERIOD_MASK                 0x00ffffff
#define LCM_RESET_CODE_WOFFSET              5
#define LCM_RESET_CODE_POS                  0
#define LCM_RESET_CODE_MASK                 0x000000ff
#define LCM_POL_FINISH_OVR_WOFFSET          5
#define LCM_POL_FINISH_OVR_POS              8
#define LCM_POL_FINISH_OVR_MASK             0x00000100
#define LCM_TP1_DONE_HIGH_WOFFSET           5
#define LCM_TP1_DONE_HIGH_POS               9
#define LCM_TP1_DONE_HIGH_MASK              0x00000200
#define LCM_AUX_CODE_EVEN_WOFFSET           6
#define LCM_AUX_CODE_EVEN_POS               0
#define LCM_AUX_CODE_EVEN_MASK              0x000000ff
#define LCM_AUX_CODE_ODD_WOFFSET            6
#define LCM_AUX_CODE_ODD_POS                8
#define LCM_AUX_CODE_ODD_MASK               0x0000ff00
#define LCM_ITO_TC_WOFFSET                  7
#define LCM_ITO_TC_POS                      0
#define LCM_ITO_TC_MASK                     0x0003ffff
#define LCM_ITO_INVERT_WOFFSET              7
#define LCM_ITO_INVERT_POS                  18
#define LCM_ITO_INVERT_MASK                 0x00040000
#define LCM_ITO_ASYNC_WOFFSET               7
#define LCM_ITO_ASYNC_POS                   19
#define LCM_ITO_ASYNC_MASK                  0x00080000
#define LCM_N_STEPS_WOFFSET                 8
#define LCM_N_STEPS_POS                     0
#define LCM_N_STEPS_MASK                    0x000000ff
#define LCM_RST_PW_WOFFSET                  8
#define LCM_RST_PW_POS                      8
#define LCM_RST_PW_MASK                     0x00000f00
#define LCM_TX_WAIT_WOFFSET                 8
#define LCM_TX_WAIT_POS                     12
#define LCM_TX_WAIT_MASK                    0x0000f000
#define LCM_TP1_PW_WOFFSET                  8
#define LCM_TP1_PW_POS                      16
#define LCM_TP1_PW_MASK                     0x00ff0000
#define LCM_POL_OVR_EN_WOFFSET              9
#define LCM_POL_OVR_EN_POS                  0
#define LCM_POL_OVR_EN_MASK                 0x00000001
#define LCM_POL_OVR_VAL_WOFFSET             9
#define LCM_POL_OVR_VAL_POS                 1
#define LCM_POL_OVR_VAL_MASK                0x00000002
#define LCM_PROG_TRIGGER_MODE_WOFFSET       10
#define LCM_PROG_TRIGGER_MODE_POS           0
#define LCM_PROG_TRIGGER_MODE_MASK          0x00000001
#define LCM_LASER_ENABLE_WOFFSET            12
#define LCM_LASER_ENABLE_POS                0
#define LCM_LASER_ENABLE_MASK               0x00000001
#define LCM_LASER_START_WOFFSET             12
#define LCM_LASER_START_POS                 0
#define LCM_LASER_START_MASK                0x00000001
#define LCM_LASER_PW_SEL_WOFFSET            13
#define LCM_LASER_PW_SEL_POS                0
#define LCM_LASER_PW_SEL_MASK               0x0000ffff
#define LCM_CLKS_PER_INTERVAL_WOFFSET       15
#define LCM_CLKS_PER_INTERVAL_POS           0
#define LCM_CLKS_PER_INTERVAL_MASK          0x000000ff
#define LCM_PULSES_PER_FRAME_WOFFSET        16
#define LCM_PULSES_PER_FRAME_POS            0
#define LCM_PULSES_PER_FRAME_MASK           0x000000ff
#define LCM_INTERVALS_PER_FRAME_WOFFSET     17
#define LCM_INTERVALS_PER_FRAME_POS         0
#define LCM_INTERVALS_PER_FRAME_MASK        0x0000ffff
#define LCM_TX_PWR_EN_WOFFSET               18
#define LCM_TX_PWR_EN_POS                   0
#define LCM_TX_PWR_EN_MASK                  0x00000001
#define LCM_TX_PWR_SWITCH_WOFFSET           18
#define LCM_TX_PWR_SWITCH_POS               1
#define LCM_TX_PWR_SWITCH_MASK              0x00000002
#define LCM_CLK_FREQ_WOFFSET                24
#define LCM_CLK_FREQ_POS                    0
#define LCM_CLK_FREQ_MASK                   0x0000ffff
#define LCM_LOADING_WOFFSET                 25
#define LCM_LOADING_POS                     0
#define LCM_LOADING_MASK                    0x00000003
#define LCM_TCON_STATE_IDLE_WOFFSET         26
#define LCM_TCON_STATE_IDLE_POS             0
#define LCM_TCON_STATE_IDLE_MASK            0x00000001
#define LCM_TCON_STATE_INIT_TP1_1H_WOFFSET  26
#define LCM_TCON_STATE_INIT_TP1_1H_POS      1
#define LCM_TCON_STATE_INIT_TP1_1H_MASK     0x00000002
#define LCM_TCON_STATE_INIT_TP1_1L_WOFFSET  26
#define LCM_TCON_STATE_INIT_TP1_1L_POS      2
#define LCM_TCON_STATE_INIT_TP1_1L_MASK     0x00000004
#define LCM_TCON_STATE_INIT_TP1_2H_WOFFSET  26
#define LCM_TCON_STATE_INIT_TP1_2H_POS      3
#define LCM_TCON_STATE_INIT_TP1_2H_MASK     0x00000008
#define LCM_TCON_STATE_INIT_TP1_2L_WOFFSET  26
#define LCM_TCON_STATE_INIT_TP1_2L_POS      4
#define LCM_TCON_STATE_INIT_TP1_2L_MASK     0x00000010
#define LCM_TCON_STATE_INIT_TP1_3H_WOFFSET  26
#define LCM_TCON_STATE_INIT_TP1_3H_POS      5
#define LCM_TCON_STATE_INIT_TP1_3H_MASK     0x00000020
#define LCM_TCON_STATE_INIT_PROG_WOFFSET    26
#define LCM_TCON_STATE_INIT_PROG_POS        6
#define LCM_TCON_STATE_INIT_PROG_MASK       0x00000040
#define LCM_TCON_STATE_INIT_TP1_4H_WOFFSET  26
#define LCM_TCON_STATE_INIT_TP1_4H_POS      7
#define LCM_TCON_STATE_INIT_TP1_4H_MASK     0x00000080
#define LCM_TCON_STATE_DONE_WOFFSET         26
#define LCM_TCON_STATE_DONE_POS             8
#define LCM_TCON_STATE_DONE_MASK            0x00000100
#define LCM_TCON_STATE_PROG_WOFFSET         26
#define LCM_TCON_STATE_PROG_POS             9
#define LCM_TCON_STATE_PROG_MASK            0x00000200
#define LCM_TCON_STATE_WAIT_WOFFSET         26
#define LCM_TCON_STATE_WAIT_POS             10
#define LCM_TCON_STATE_WAIT_MASK            0x00000400
#define LCM_TCON_STATE_POL_WOFFSET          26
#define LCM_TCON_STATE_POL_POS              11
#define LCM_TCON_STATE_POL_MASK             0x00000800
#define LCM_TCON_STATE_TP1_WOFFSET          26
#define LCM_TCON_STATE_TP1_POS              12
#define LCM_TCON_STATE_TP1_MASK             0x00001000
#define LCM_TCON_STATE_PROG_FIN_WOFFSET     26
#define LCM_TCON_STATE_PROG_FIN_POS         13
#define LCM_TCON_STATE_PROG_FIN_MASK        0x00002000
#define LCM_TCON_STATE_WAIT_FIN_WOFFSET     26
#define LCM_TCON_STATE_WAIT_FIN_POS         14
#define LCM_TCON_STATE_WAIT_FIN_MASK        0x00004000
#define LCM_TCON_STATE_POL_FIN_WOFFSET      26
#define LCM_TCON_STATE_POL_FIN_POS          15
#define LCM_TCON_STATE_POL_FIN_MASK         0x00008000
#define LCM_TCON_STATE_TP1_FIN_WOFFSET      26
#define LCM_TCON_STATE_TP1_FIN_POS          16
#define LCM_TCON_STATE_TP1_FIN_MASK         0x00010000
#define LCM_APPLY_CACHE_WOFFSET             27
#define LCM_APPLY_CACHE_POS                 0
#define LCM_APPLY_CACHE_MASK                0x00000003

//  - SPI registers/fields
#define SPI_RX_SWITCH_WOFFSET               0
#define SPI_RX_SWITCH_POS                   0
#define SPI_RX_SWITCH_MASK                  0x0000ffff
#define SPI_TX_SWITCH_WOFFSET               8
#define SPI_TX_SWITCH_POS                   0
#define SPI_TX_SWITCH_MASK                  0x0000ffff
#define SPI_CLK_DIV_ADC_WOFFSET             24
#define SPI_CLK_DIV_ADC_POS                 0
#define SPI_CLK_DIV_ADC_MASK                0x0000001f
#define SPI_CLK_DIV_SWITCH_WOFFSET          24
#define SPI_CLK_DIV_SWITCH_POS              5
#define SPI_CLK_DIV_SWITCH_MASK             0x000003e0
#define SPI_CLK_DIV_TMP_WOFFSET             24
#define SPI_CLK_DIV_TMP_POS                 10
#define SPI_CLK_DIV_TMP_MASK                0x00007c00
#define SPI_CLK_DIV_POT_ITO_WOFFSET         24
#define SPI_CLK_DIV_POT_ITO_POS             15
#define SPI_CLK_DIV_POT_ITO_MASK            0x000f8000
#define SPI_CLK_DIV_POT_TX_WOFFSET          24
#define SPI_CLK_DIV_POT_TX_POS              20
#define SPI_CLK_DIV_POT_TX_MASK             0x01f00000
#define SPI_FILL_CMD_FIFO_WOFFSET           25
#define SPI_FILL_CMD_FIFO_POS               0
#define SPI_FILL_CMD_FIFO_MASK              0x00000001
#define SPI_DAISY_EN_WOFFSET                25
#define SPI_DAISY_EN_POS                    1
#define SPI_DAISY_EN_MASK                   0x00000002
#define SPI_ADC_CH_SEL_WOFFSET              25
#define SPI_ADC_CH_SEL_POS                  2
#define SPI_ADC_CH_SEL_MASK                 0x0000000c
#define SPI_CMD_FIFO_WOFFSET                26
#define SPI_CMD_FIFO_POS                    0
#define SPI_CMD_FIFO_MASK                   0xffffffff
#define SPI_RSP_FIFO_1_WOFFSET              29
#define SPI_RSP_FIFO_1_POS                  0
#define SPI_RSP_FIFO_1_MASK                 0xffffffff
#define SPI_RSP_FIFO_0_WOFFSET              30
#define SPI_RSP_FIFO_0_POS                  0
#define SPI_RSP_FIFO_0_MASK                 0xffffffff
#define SPI_CMD_FIFO_COUNT_WOFFSET          31
#define SPI_CMD_FIFO_COUNT_POS              0
#define SPI_CMD_FIFO_COUNT_MASK             0x0000000f
#define SPI_CMD_FIFO_FULL_WOFFSET           31
#define SPI_CMD_FIFO_FULL_POS               4
#define SPI_CMD_FIFO_FULL_MASK              0x00000010


// AXI4-Lite Peek-Poker
struct axi4lpp{
    int dev_mem_fd;
    uint32_t *base_ddr;
    uint32_t *base_cdma;
    uint32_t *base_lcm;
    uint32_t *base_spi;
    float lcm_clk_freq_MHz;
    uint32_t pol_ovr;
};

// Init
int axi4lpp_init(struct axi4lpp *pp);
int _axi4lpp_open_dev_mem(int *dev_mem_fd);
int _axi4lpp_make_map(off_t dev_base_addr,
                      size_t map_size,
                      uint32_t **mapped_base_addr,
                      int *dev_mem_fd);
int _axi4lpp_configure_ps7(struct axi4lpp *pp);
int axi4lpp_set_fabric_reset(struct axi4lpp *pp, uint32_t reset_en);
int axi4lpp_read_pattern_file(struct axi4lpp *pp, const char *filename);
void axi4lpp_dump(struct axi4lpp *pp, FILE *fptr);

// Delete
void axi4lpp_del(struct axi4lpp *pp);
void _axi4lpp_close_map(size_t map_size, uint32_t **mapped_base_addr);
void _axi4lpp_close_dev_mem(int *dev_mem_fd);
void axi4lpp_shutdown(struct axi4lpp *pp);

// AXI4-Lite read and write
uint32_t axi4lpp_axi_read(uint32_t *base_addr, uint32_t word_addr);
void axi4lpp_axi_write(uint32_t *base_addr, uint32_t word_addr, uint32_t data);
uint32_t axi4lpp_get_field(uint32_t *base_addr,
                           uint32_t word_addr,
                           uint32_t pos,
                           uint32_t mask);
void axi4lpp_set_field(uint32_t *base_addr,
                       uint32_t word_addr,
                       uint32_t pos,
                       uint32_t mask,
                       uint32_t value);

// CDMA
void cdma_reset(struct axi4lpp *pp);
int cdma_reset_is_done(struct axi4lpp *pp);
int cdma_reset_and_wait(struct axi4lpp *pp, int tries);
int cdma_is_busy(struct axi4lpp *pp);
int cdma_has_error(struct axi4lpp *pp);
uint32_t cdma_get_ddr_addr(struct axi4lpp *pp, uint32_t table_idx);
uint32_t cdma_get_bram_addr(struct axi4lpp *pp, uint32_t pingpong_idx);
int cdma_do_transfer(struct axi4lpp *pp,
                     uint32_t src_addr,
                     uint32_t dst_addr,
                     uint32_t n_bytes);
int cdma_transfer_down(struct axi4lpp *pp,
                       uint32_t ddr_idx,
                       uint32_t bram_idx);
int cdma_transfer_up(struct axi4lpp *pp,
                     uint32_t ddr_idx,
                     uint32_t bram_idx);

// LCM
int lcm_buf_is_loading(struct axi4lpp *pp, uint32_t bram_idx);
int lcm_buf_is_queued(struct axi4lpp *pp, uint32_t bram_idx);

#endif
