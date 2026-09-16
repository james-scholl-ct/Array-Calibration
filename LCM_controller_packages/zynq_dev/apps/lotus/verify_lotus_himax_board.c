#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#include "lotus_driver.h"
#include "lotus_driver_bsc.h"
#include "lotus_driver_spi.h"


// Maps integral voltages 0-9 to Himax codes, assuming odd channels and POL low.
const uint8_t V_MAP[10] = {0xff, 0xf7, 0xeb, 0xd9, 0xbc, 0x91, 0x49, 0x0b, 0x04, 0x00};

void wait_us(int us)
{
    // OS can't reliably support sleep < 100 us
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = us * 1000;
    nanosleep(&ts, NULL);
}


float sample_channel_voltage(struct axi4lpp *pp, int channel)
{
    uint32_t rsp;
    int test;
    uint32_t data;

    axi4lpp_spi_send_cmd(pp, STANDARD, ADC1, 0x0000);
    axi4lpp_spi_send_cmd(pp, STANDARD, ADC1, 0x0000);
    do {
        rsp = axi4lpp_spi_read_rsp(pp);
    } while (! axi4lpp_spi_get_rsp_is_valid(pp, rsp));
    do {
        rsp = axi4lpp_spi_read_rsp(pp);
    } while (! axi4lpp_spi_get_rsp_is_valid(pp, rsp));

    test = axi4lpp_spi_get_rsp_is_valid(pp, rsp);
    test &= axi4lpp_spi_get_rsp_slave_idx(pp, rsp) == ADC1;
    test &= axi4lpp_spi_get_rsp_fifo_count(pp, rsp) == 1;
    if (!test)
    {
        printf("ERROR: Failed to sample channel %d\n", channel);
    }

    data = axi4lpp_spi_get_rsp_payload(pp, rsp) >> 2;
    return (float)(data * 0.00025 * 1.5);
}


void select_channel(struct axi4lpp *pp, int channel)
{
    int word_addr = LOTUS_SPI_SWITCH_MAP[channel][0];
    int bit = LOTUS_SPI_SWITCH_MAP[channel][1];
    uint32_t vt_words[7];
    uint32_t gnd_words[7];
    for (int i = 0; i < 7; i = i + 1)
    {
        if (word_addr == i)
        {
            vt_words[i] = 1 << bit;
            gnd_words[i] = 0xffffffff & ~(1 << bit);
        }
        else
        {
            vt_words[i] = 0;
            gnd_words[i] = 0xffffffff;
        }
    }
    axi4lpp_spi_write_jumbo_frame_data(pp, gnd_words, vt_words);
    axi4lpp_spi_send_cmd(pp, JUMBO, DAISY, 12);
}


void enable_daisy_chain_mode(struct axi4lpp *pp)
{
    axi4lpp_spi_send_cmd(pp, STANDARD, DAISY, 0x2500);
    uint32_t rsp;
    do {
        rsp = axi4lpp_spi_read_rsp(pp);
    } while (! axi4lpp_spi_get_rsp_is_valid(pp, rsp));
}


void reset_switches(struct axi4lpp *pp)
{
    axi4lpp_spi_set_config(pp, 0x0);
    wait_us(500);
    axi4lpp_spi_set_config(pp, 0x4);
    wait_us(500);
}


void drain_rsp_fifo(struct axi4lpp *pp)
{
    uint32_t rsp;
    do {
        rsp = axi4lpp_spi_read_rsp(pp);
    } while (axi4lpp_spi_get_rsp_is_valid(pp, rsp));
}


void init_lotus(struct axi4lpp *pp)
{
    pp->pol_ovr = 1;
    axi4lpp_bsc_init(pp);
    axi4lpp_bsc_set_dwell(pp, axi4lpp_bsc_dwell_cnt_from_us(pp, 10.));
    axi4lpp_spi_config_clkdivs(pp, 0x00022292);
    drain_rsp_fifo(pp);
    reset_switches(pp);
    enable_daisy_chain_mode(pp);
    assert(axi4lpp_bsc_driver_is_done(pp));
}


void sweep_channels(struct axi4lpp *pp)
{
    // open log file
    FILE *fp_log;
    const char *logfile = "verify_lotus_himax_board.log";
    fp_log = fopen(logfile, "w");
    if (!fp_log)
    {
        printf("ERROR: can't open file %s.\n", logfile);
        return;
    }

    // init table to 0V for all channels (channels are odd and POL=0)
    uint8_t table[204];
    for (int i = 0; i < 204; i = i + 1)
    {
        table[i] = V_MAP[0];
    }

    init_lotus(pp);
    for (int i = 0; i < 204; i = i + 1)
    {
        table[i] = V_MAP[2];
        axi4lpp_bsc_set_table(pp, table);
        axi4lpp_bsc_apply_table(pp);
        for (int j = 0; j < 204; j = j + 1)
        {
            select_channel(pp, j);
            wait_us(100);
            float voltage = sample_channel_voltage(pp, j);
            fprintf(fp_log, "%d, %d, %.3f\n", i, j, voltage);
        }
        table[i] = V_MAP[0];
    }
}


int main(int argc, char* argv[])
{
    char buf[80];
    printf("\nPlease remove all jumpers on the short-check board.\n");
    printf("Proceed? [y/n] ");
    scanf("%s", buf);
    if (strcmp(buf, "y") == 0)
    {
        struct axi4lpp *pp = (struct axi4lpp *)malloc(sizeof(struct axi4lpp));
        axi4lpp_init(pp);

        sweep_channels(pp);

        // pol_ovr attribute will be applied during axi4lpp_bsc_stop
        pp->pol_ovr = 0;
        axi4lpp_bsc_stop(pp);
        axi4lpp_spi_set_config(pp, 0x0);
        axi4lpp_del(pp);
        free(pp);
    }
    return 0;
}
