#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#include "delorean_driver.h"


struct axi4lpp *pp;


void handle_sigint(int sig)
{
    fprintf(stdout, "\nCaught CTRL-C.\n");
    axi4lpp_shutdown(pp);
    axi4lpp_del(pp);
    free(pp);
    exit(0);
}


void configure_defaults(struct axi4lpp *pp)
{
    // TP1: 20 us, 50 kHz (POL: 40 us, 25 kHz)
    axi4lpp_set_field(pp->base_lcm, LCM_TP1_PERIOD_WOFFSET,
                                    LCM_TP1_PERIOD_POS,
                                    LCM_TP1_PERIOD_MASK,
                                    1996);
    // 171 steps = 1026 channels / (6 channels / step)
    axi4lpp_set_field(pp->base_lcm, LCM_N_STEPS_WOFFSET,
                                    LCM_N_STEPS_POS,
                                    LCM_N_STEPS_MASK,
                                    170);
    // RST is 5 LVDS_CLK_P cycles on LVDS_DATA[0]
    axi4lpp_set_field(pp->base_lcm, LCM_RST_PW_WOFFSET,
                                    LCM_RST_PW_POS,
                                    LCM_RST_PW_MASK,
                                    4);
    // At least 11 cycles between 1) LVDS_CLK_P of last data and 2) TP1 posedge
    axi4lpp_set_field(pp->base_lcm, LCM_TX_WAIT_WOFFSET,
                                    LCM_TX_WAIT_POS,
                                    LCM_TX_WAIT_MASK,
                                    7);
    // TP1 high pulse width is 50 clock cycles (500 ns)
    axi4lpp_set_field(pp->base_lcm, LCM_TP1_PW_WOFFSET,
                                    LCM_TP1_PW_POS,
                                    LCM_TP1_PW_MASK,
                                    49);
    // PROG_TRIGGER - 0: toggle mode, 1: pulse mode
    axi4lpp_set_field(pp->base_lcm, LCM_PROG_TRIGGER_MODE_WOFFSET,
                                    LCM_PROG_TRIGGER_MODE_POS,
                                    LCM_PROG_TRIGGER_MODE_MASK,
                                    1);
}


int main(int argc, char* argv[])
{
    signal(SIGINT, handle_sigint);

    int status;
    uint32_t data;
    uint32_t idx_a;
    uint32_t idx_b;

    // Parse args
    if(argc != 4)
    {
        fprintf(stderr, "You must specify the pattern file and two angle "
                        "indices on the command line (0 <= idx < 1024).\n");
        exit(-1);
    }
    idx_a = (uint32_t)strtol(argv[2], NULL, 10);
    idx_b = (uint32_t)strtol(argv[3], NULL, 10);
    if (idx_a < 0 || idx_a >= 1024)
    {
        fprintf(stderr, "ERROR: invalid value idx_a = %d.\n", idx_a);
        exit(-1);
    }
    if (idx_b < 0 || idx_b >= 1024)
    {
        fprintf(stderr, "ERROR: invalid value idx_b = %d.\n", idx_b);
        exit(-1);
    }
    fprintf(stdout, "Parsed idx_a = %d, idx_b = %d.\n", idx_a, idx_b);

    // Init AXI4-Lite peek-poker
    pp = (struct axi4lpp *)malloc(sizeof(struct axi4lpp));
    status = axi4lpp_init(pp);
    if (status == FAIL)
    {
        fprintf(stderr, "ERROR: axi4lpp_init failed.\n");
        axi4lpp_dump(pp, stderr);
        free(pp);
        exit(-1);
    }
    fprintf(stdout, "Created axi4lpp object.\n");

    // Configure defaults for LCM peripheral
    configure_defaults(pp);
    fprintf(stdout, "Configured LCM peripheral with defaults.\n");

    // Reset CDMA
    status = cdma_reset_and_wait(pp, 10);
    if (status == FAIL)
    {
        fprintf(stderr, "ERROR: cdma_reset_and_wait failed.\n");
        axi4lpp_del(pp);
        free(pp);
        exit(-1);
    }
    fprintf(stdout, "CDMA peripheral reset.\n");

    // Init DDR space with pattern memory
    status = axi4lpp_read_pattern_file(pp, argv[1]);
    if (status == FAIL)
    {
        fprintf(stderr, "ERROR: axi4lpp_read_pattern_file failed.\n");
        axi4lpp_del(pp);
        free(pp);
        exit(-1);
    }
    fprintf(stdout, "Initialized DDR with patterns from file %s.\n", argv[1]);

    // Start the TCON FSM
    axi4lpp_axi_write(pp->base_lcm, LCM_TCON_ENABLE_WOFFSET,
                                    LCM_TCON_ENABLE_MASK);
    do{
        data = axi4lpp_axi_read(pp->base_lcm, LCM_TCON_STATE_DONE_WOFFSET);
        // wait for DONE state
    } while (!(data & LCM_TCON_STATE_DONE_MASK));
    fprintf(stdout, "LCM TCON initialized.\n");

    // Run ping-pong
    fprintf(stdout, "Started ping-pong on tables %d and %d.\n", idx_a, idx_b);
    int cnt;
    while (1)
    {
        // Transfer Table A to Buf 0, test Buf 1, then apply Buf 0
        status = cdma_transfer_down(pp, idx_a, 0);
        if (status == FAIL)
        {
            fprintf(stderr, "ERROR: cdma_transfer_down failed (Buf 0).\n");
            axi4lpp_shutdown(pp);
            axi4lpp_del(pp);
            free(pp);
            exit(-1);
        }
        cnt = 0;
        while (lcm_buf_is_queued(pp, 1) || lcm_buf_is_loading(pp, 1))
        {
            // wait for buf 1 to be applied (nop on first loop).
            fprintf(stdout, "Waiting on buf 1: %d.\n", cnt++);
        }
        axi4lpp_axi_write(pp->base_lcm, LCM_APPLY0_WOFFSET, 0);

        // Transfer Table B to Buf 1, test Buf 0, then apply Buf 1
        cdma_transfer_down(pp, idx_b, 1);
        if (status == FAIL)
        {
            fprintf(stderr, "ERROR: cdma_transfer_down failed (Buf 1).\n");
            axi4lpp_shutdown(pp);
            axi4lpp_del(pp);
            free(pp);
            exit(-1);
        }
        cnt = 0;
        while (lcm_buf_is_queued(pp, 0) || lcm_buf_is_loading(pp, 0))
        {
            // wait for buf 0 to be applied/programmed
            fprintf(stdout, "Waiting on buf 0: %d.\n", cnt++);
        }
        axi4lpp_axi_write(pp->base_lcm, LCM_APPLY1_WOFFSET, 0);
    }
}
