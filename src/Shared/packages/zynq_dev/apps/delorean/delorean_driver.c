#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "delorean_driver.h"


// ----------------------------------------------------------------------------
// Init
// ----------------------------------------------------------------------------
int axi4lpp_init(struct axi4lpp *pp)
{
    int status;

    // Open the /dev/mem file to access hardware space
    status = _axi4lpp_open_dev_mem(&pp->dev_mem_fd);
    if (status == FAIL)
    {
        return FAIL;
    }

    // Map DDR space
    status = _axi4lpp_make_map(HP0_DDR_BASEADDR,
                               HP0_DDR_MAP_SIZE,
                               &pp->base_ddr,
                               &pp->dev_mem_fd);
    if (status == FAIL)
    {
        _axi4lpp_close_dev_mem(&pp->dev_mem_fd);
        return FAIL;
    }

    // Map CDMA space
    status = _axi4lpp_make_map(CDMA_BASEADDR,
                               CDMA_MAP_SIZE,
                               &pp->base_cdma,
                               &pp->dev_mem_fd);
    if (status == FAIL)
    {
        _axi4lpp_close_map(HP0_DDR_MAP_SIZE, &pp->base_ddr);
        _axi4lpp_close_dev_mem(&pp->dev_mem_fd);
        return FAIL;
    }

    // Map LCM space
    status = _axi4lpp_make_map(LCM_BASEADDR,
                               LCM_MAP_SIZE,
                               &pp->base_lcm,
                               &pp->dev_mem_fd);
    if (status == FAIL)
    {
        _axi4lpp_close_map(HP0_DDR_MAP_SIZE, &pp->base_ddr);
        _axi4lpp_close_map(CDMA_MAP_SIZE, &pp->base_cdma);
        _axi4lpp_close_dev_mem(&pp->dev_mem_fd);
        return FAIL;
    }

    // Map SPI space
    status = _axi4lpp_make_map(SPI_BASEADDR,
                               SPI_MAP_SIZE,
                               &pp->base_spi,
                               &pp->dev_mem_fd);
    if (status == FAIL)
    {
        _axi4lpp_close_map(HP0_DDR_MAP_SIZE, &pp->base_ddr);
        _axi4lpp_close_map(CDMA_MAP_SIZE, &pp->base_cdma);
        _axi4lpp_close_map(LCM_MAP_SIZE, &pp->base_lcm);
        _axi4lpp_close_dev_mem(&pp->dev_mem_fd);
        return FAIL;
    }

    // Configure Zynq-7000 processing system (PS7)
    status = _axi4lpp_configure_ps7(pp);
    if (status == FAIL)
    {
        axi4lpp_del(pp);
        return FAIL;
    }

    // Init object attributes
    pp->lcm_clk_freq_MHz = 100.;
    pp->pol_ovr = 0;

    return SUCCESS;
}


int _axi4lpp_open_dev_mem(int *dev_mem_fd)
{
    // need root access to open this file
    *dev_mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (*dev_mem_fd == -1)
    {
        fprintf(stderr, "ERROR: can't open /dev/mem.\n");
        return FAIL;
    }
    return SUCCESS;
}


int _axi4lpp_make_map(off_t dev_base_addr,
                      size_t map_size,
                      uint32_t **mapped_base_addr,
                      int *dev_mem_fd)
{
    *mapped_base_addr = mmap(0,
                             map_size,
                             PROT_READ | PROT_WRITE,
                             MAP_SHARED,
                             *dev_mem_fd,
                             dev_base_addr);
    if (*mapped_base_addr == (uint32_t *) -1)
    {
        fprintf(stderr, "ERROR: can't map mem to 0x%x.\n",
                        (unsigned int)dev_base_addr);
        return FAIL;
    }
    return SUCCESS;
}


int _axi4lpp_configure_ps7(struct axi4lpp *pp)
{
    int status;
    volatile uint32_t *addr;

    // Configure HP0 for 32-bit mode.
    //  - See UG585: Zynq-7000 SoC Technical Reference Manual
    uint32_t *base_afi_hp0;
    status = _axi4lpp_make_map(0xF8008000,
                               4096,
                               &base_afi_hp0,
                               &pp->dev_mem_fd);
    if (status == FAIL)
    {
        return FAIL;
    }

    //  - HP0 AFI_RDCHAN_CTRL.32BitEn
    addr = base_afi_hp0 + 0;
    *addr = 0x1;
    //  - HP0 AFI_WRCHAN_CTRL.32BitEn
    addr = base_afi_hp0 + 5;
    *addr = 0x1;

    _axi4lpp_close_map(4096, &base_afi_hp0);

    // TODO: Add LVL_SHFTR_EN config 0xF8000900, bits 3:0
    //

    return SUCCESS;
}


int axi4lpp_set_fabric_reset(struct axi4lpp *pp, uint32_t reset_en)
{
    int status;
    volatile uint32_t *addr;

    // Reset fabric
    //  - See UG585: Zynq-7000 SoC Technical Reference Manual
    uint32_t *base_slcr;
    status = _axi4lpp_make_map(0xF8000000,
                               4096,
                               &base_slcr,
                               &pp->dev_mem_fd);
    if (status == FAIL)
    {
        return FAIL;
    }

    //  - FPGA_RST_CTRL[3:0]
    //  - The register is active-high but the PS7 pin is active-low
    addr = base_slcr + 144;
    *addr = (reset_en) ? 0xf : 0x0;

    _axi4lpp_close_map(4096, &base_slcr);
    return SUCCESS;
}


int axi4lpp_read_pattern_file(struct axi4lpp *pp, const char *filename)
{
    FILE *fptr = fopen(filename, "r");
    if (fptr == NULL)
    {
        fprintf(stderr, "ERROR: Cannot open pattern file %s.\n", filename);
        return FAIL;
    }

    // scan for a hex number at the start of each line and ignore rest of line
    int data;
    uint32_t i = 0;
    while(fscanf(fptr, "%x%*[^\n]\n", &data) > 0)
    {
        // this is not actually an AXI transaction; leverage mmap call
        axi4lpp_axi_write(pp->base_ddr, i, (uint32_t)data);
        i += 1;
    }
    fclose(fptr);

    if (i != HP0_DDR_MAP_SIZE / DATAPATH_BYTESIZE)
    {
        fprintf(stderr, "ERROR: Incorrect number of words parsed (%d).\n", i);
        return FAIL;
    }
    return SUCCESS;
}


void axi4lpp_dump(struct axi4lpp *pp, FILE *fptr)
{
    fprintf(fptr, "Mapped DDR space to 0x%x.\n", (unsigned int)pp->base_ddr);
    fprintf(fptr, "Mapped CDMA space to 0x%x.\n", (unsigned int)pp->base_cdma);
    fprintf(fptr, "Mapped LCM space to 0x%x.\n", (unsigned int)pp->base_lcm);
    fprintf(fptr, "Mapped SPI space to 0x%x.\n", (unsigned int)pp->base_spi);
}


// ----------------------------------------------------------------------------
// Delete
// ----------------------------------------------------------------------------
void axi4lpp_del(struct axi4lpp *pp)
{
    _axi4lpp_close_map(HP0_DDR_MAP_SIZE, &pp->base_ddr);
    _axi4lpp_close_map(CDMA_MAP_SIZE, &pp->base_cdma);
    _axi4lpp_close_map(LCM_MAP_SIZE, &pp->base_lcm);
    _axi4lpp_close_map(SPI_MAP_SIZE, &pp->base_spi);
    _axi4lpp_close_dev_mem(&pp->dev_mem_fd);
}


void _axi4lpp_close_map(size_t map_size, uint32_t **mapped_base_addr)
{
    int status = munmap(*mapped_base_addr, map_size);
    if (status == -1)
    {
        fprintf(stderr, "ERROR: couldn't unmap memory from user space.\n");
    }
}


void _axi4lpp_close_dev_mem(int *dev_mem_fd)
{
    close(*dev_mem_fd);
}


void axi4lpp_shutdown(struct axi4lpp *pp)
{
    axi4lpp_axi_write(pp->base_lcm, LCM_LASER_ENABLE_WOFFSET, 0);
    axi4lpp_axi_write(pp->base_lcm, LCM_TX_PWR_EN_WOFFSET, 0);
    axi4lpp_axi_write(pp->base_lcm, LCM_TCON_ENABLE_WOFFSET, 0);
    uint32_t data;
    uint32_t done = 0;
    while (!done)
    {
        data = axi4lpp_axi_read(pp->base_lcm, LCM_TCON_STATE_IDLE_WOFFSET);
        // wait for IDLE state or DONE state
        done = (data & LCM_TCON_STATE_IDLE_MASK) |
               (data & LCM_TCON_STATE_DONE_MASK);
    }
    axi4lpp_axi_write(pp->base_lcm, LCM_LCD_EN_WOFFSET, 0);
    axi4lpp_set_fabric_reset(pp, 1);
    axi4lpp_set_fabric_reset(pp, 0);
}


// ----------------------------------------------------------------------------
// AXI4-Lite read and write
// ----------------------------------------------------------------------------
uint32_t axi4lpp_axi_read(uint32_t *base_addr, uint32_t word_addr)
{
    volatile uint32_t *addr = base_addr + word_addr;
    return *addr;
}


void axi4lpp_axi_write(uint32_t *base_addr, uint32_t word_addr, uint32_t data)
{
    volatile uint32_t *addr = base_addr + word_addr;
    *addr = data;
}


uint32_t axi4lpp_get_field(uint32_t *base_addr,
                           uint32_t word_addr,
                           uint32_t pos,
                           uint32_t mask)
{
    uint32_t data = axi4lpp_axi_read(base_addr, word_addr);
    return (data & mask) >> pos;
}


void axi4lpp_set_field(uint32_t *base_addr,
                       uint32_t word_addr,
                       uint32_t pos,
                       uint32_t mask,
                       uint32_t value)
{
    uint32_t data = axi4lpp_axi_read(base_addr, word_addr);
    data &= ~mask;
    data |= (value << pos) & mask;
    axi4lpp_axi_write(base_addr, word_addr, data);
}


// ----------------------------------------------------------------------------
// CDMA
// ----------------------------------------------------------------------------
void cdma_reset(struct axi4lpp *pp)
{
    axi4lpp_axi_write(pp->base_cdma,
                      CDMA_CDMACR_WOFFSET,
                      CDMA_CDMACR_RESET_MASK);
    return;
}


int cdma_reset_is_done(struct axi4lpp *pp)
{
    // If the reset bit is still high, then reset is not done
    return ((axi4lpp_axi_read(pp->base_cdma, CDMA_CDMACR_WOFFSET) &
             CDMA_CDMACR_RESET_MASK) ? 0 : 1);
}


int cdma_reset_and_wait(struct axi4lpp *pp, int tries)
{
    cdma_reset(pp);
    int i;
    for (i = 0; i < tries; i += 1)
    {
        if (cdma_reset_is_done(pp))
        {
            return SUCCESS;
        }
    }
    fprintf(stderr, "ERROR: reset failed.\n");
    return FAIL;
}


int cdma_is_busy(struct axi4lpp *pp)
{
    // If the IDLE bit is high, then not busy
    return ((axi4lpp_axi_read(pp->base_cdma, CDMA_CDMASR_WOFFSET) &
             CDMA_CDMASR_IDLE_MASK) ? 0 : 1);
}


int cdma_has_error(struct axi4lpp *pp)
{
    // If any error flag is asserted, then has error
    return ((axi4lpp_axi_read(pp->base_cdma, CDMA_CDMASR_WOFFSET) &
             CDMA_CDMASR_ERR_ALL_MASK) ? 1 : 0);
}


uint32_t cdma_get_ddr_addr(struct axi4lpp *pp, uint32_t table_idx)
{
    // The physical address corresponds to bytes.
    return (uint32_t)(HP0_DDR_BASEADDR + table_idx * DMA_BTT_MAX);
}


uint32_t cdma_get_bram_addr(struct axi4lpp *pp, uint32_t pingpong_idx)
{
    // The physical address corresponds to bytes.
    return (uint32_t)(BRAM_CTRL_BASEADDR + pingpong_idx * DMA_BTT_MAX);
}


int cdma_do_transfer(struct axi4lpp *pp,
                     uint32_t src_addr,
                     uint32_t dst_addr,
                     uint32_t n_bytes)
{
    int status;

    uint32_t byte_sel_mask = DATAPATH_BYTESIZE - 1;
    if ((src_addr & byte_sel_mask) || (dst_addr & byte_sel_mask))
    {
        fprintf(stderr, "ERROR: Unaligned transfer.\n");
        return FAIL;
    }

    // If the engine is already transferring, don't interrupt
    if (cdma_is_busy(pp))
    {
        fprintf(stderr, "ERROR: Engine is busy.\n");
        return FAIL;
    }

    // Writing to the BTT register starts the transfer
    axi4lpp_axi_write(pp->base_cdma, CDMA_SA_WOFFSET, src_addr);
    axi4lpp_axi_write(pp->base_cdma, CDMA_DA_WOFFSET, dst_addr);
    axi4lpp_axi_write(pp->base_cdma, CDMA_BTT_WOFFSET, n_bytes);

    while (cdma_is_busy(pp))
    {
        // Wait
    }

    if (cdma_has_error(pp))
    {
        // If the hardware has errors, reset to clear them
        fprintf(stderr, "ERROR: cdma encountered an error.\n");
        status = cdma_reset_and_wait(pp, 10);
        if (status == FAIL)
        {
            fprintf(stderr, "ERROR: reset failed after error condition.\n");
        }
        return FAIL;
    }
    return SUCCESS;
}


int cdma_transfer_down(struct axi4lpp *pp, uint32_t ddr_idx, uint32_t bram_idx)
{
    int status;
    if (ddr_idx >= LCM_MAX_TABLES)
    {
        fprintf(stderr, "ERROR: invalid ddr_idx %d.\n", ddr_idx);
        return FAIL;
    }
    if (bram_idx >= 2)
    {
        fprintf(stderr, "ERROR: invalid bram_idx %d.\n", bram_idx);
        return FAIL;
    }
    uint32_t src_addr = cdma_get_ddr_addr(pp, ddr_idx);
    uint32_t dst_addr = cdma_get_bram_addr(pp, bram_idx);
    if (lcm_buf_is_queued(pp, bram_idx) ||
        lcm_buf_is_loading(pp, bram_idx))
    {
        fprintf(stderr, "ERROR: Buffer is currently in use for down "
                        "transfer from ddr_idx = %d to bram_idx = %d.\n",
                        ddr_idx,
                        bram_idx);
        return FAIL;
    }
    status = cdma_do_transfer(pp, src_addr, dst_addr, DMA_BTT_MAX);
    return status;
}


int cdma_transfer_up(struct axi4lpp *pp, uint32_t ddr_idx, uint32_t bram_idx)
{
    int status;
    if (ddr_idx >= LCM_MAX_TABLES)
    {
        fprintf(stderr, "ERROR: invalid ddr_idx %d.\n", ddr_idx);
        return FAIL;
    }
    if (bram_idx >= 2)
    {
        fprintf(stderr, "ERROR: invalid bram_idx %d.\n", bram_idx);
        return FAIL;
    }
    uint32_t src_addr = cdma_get_bram_addr(pp, bram_idx);
    uint32_t dst_addr = cdma_get_ddr_addr(pp, ddr_idx);
    status = cdma_do_transfer(pp, src_addr, dst_addr, DMA_BTT_MAX);
    return status;
}


// ----------------------------------------------------------------------------
// LCM
// ----------------------------------------------------------------------------
int lcm_buf_is_loading(struct axi4lpp *pp, uint32_t bram_idx)
{
    uint32_t loading;
    loading = axi4lpp_axi_read(pp->base_lcm, LCM_LOADING_WOFFSET);
    return (loading & (1 << bram_idx) ? 1 : 0);
}


int lcm_buf_is_queued(struct axi4lpp *pp, uint32_t bram_idx)
{
    uint32_t apply_cache;
    apply_cache = axi4lpp_axi_read(pp->base_lcm, LCM_APPLY_CACHE_WOFFSET);
    return (apply_cache & (1 << bram_idx) ? 1 : 0);
}
