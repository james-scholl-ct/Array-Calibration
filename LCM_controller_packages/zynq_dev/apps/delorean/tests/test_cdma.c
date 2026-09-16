#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "delorean_driver.h"

int test_cdma(int verbose, int ddr_idx_from, int ddr_idx_to, int bram_idx)
{
    int status;
    int i;
    struct axi4lpp *pp;

    if (verbose)
        printf("\r\n--- Entering main() ---\r\n");

    if (verbose)
        printf("\r\n--- Setting up user maps ---\r\n");
    pp = (struct axi4lpp *)malloc(sizeof(struct axi4lpp));
    status = axi4lpp_init(pp);
    if (status == FAIL)
    {
        printf("axi4lpp_init FAILED.\n");
        axi4lpp_del(pp);
        free(pp);
        return 1;
    }
    if (verbose)
        axi4lpp_dump(pp, stdout);

    if (verbose)
        printf("\r\n--- Reading pattern file ---\r\n");
    status = axi4lpp_read_pattern_file(pp, "../patterns.txt");
    if (status == FAIL)
    {
        printf("axi4lpp_read_pattern_file FAILED.\n");
        axi4lpp_del(pp);
        free(pp);
        return 1;
    }

    if (verbose)
        printf("\r\n--- Resetting CDMA ---\r\n");
    status = cdma_reset_and_wait(pp, 10);
    if (status == FAIL)
    {
        printf("cdma_reset_and_wait FAILED.\n");
        axi4lpp_del(pp);
        free(pp);
        return 1;
    }

    if (verbose)
        printf("\r\n--- Checking DDR tables for INEQUALITY. ---\r\n");
    status = 0;
    for (i = 0; i < 512; i += 1)
    {
        uint32_t rdata_a = axi4lpp_axi_read(pp->base_ddr, i+512*ddr_idx_from);
        uint32_t rdata_b = axi4lpp_axi_read(pp->base_ddr, i+512*ddr_idx_to);
        if (verbose && (i % 512 < 10 || i % 512 > 502))
        {
            printf("Read %08x / %08x from DDR word %d-%d / %d-%d.\n",
                   rdata_a, rdata_b, ddr_idx_from, i, ddr_idx_to, i);
        }
        status = status || (rdata_a == rdata_b);
    }
    if (status)
    {
        printf("ERROR: Tables are not distinct at init time.\n");
        axi4lpp_del(pp);
        free(pp);
        return 1;
    }

    if (verbose)
        printf("\r\n--- Running CDMA ---\r\n");
    status = cdma_transfer_down(pp, ddr_idx_from, bram_idx);
    if (status == FAIL)
    {
        printf("cdma_transfer_down FAILED.\n");
        axi4lpp_del(pp);
        free(pp);
        return 1;
    }
    status = cdma_transfer_up(pp, ddr_idx_to, bram_idx);
    if (status == FAIL)
    {
        printf("cdma_transfer_up FAILED.\n");
        axi4lpp_del(pp);
        free(pp);
        return 1;
    }

    if (verbose)
        printf("\r\n--- Checking DDR tables for EQUALITY. ---\r\n");
    status = 0;
    for (i = 0; i < 512; i += 1)
    {
        uint32_t rdata_a = axi4lpp_axi_read(pp->base_ddr, i+512*ddr_idx_from);
        uint32_t rdata_b = axi4lpp_axi_read(pp->base_ddr, i+512*ddr_idx_to);
        if (verbose && (i % 512 < 10 || i % 512 > 502))
        {
            printf("Read %08x / %08x from DDR word %d-%d / %d-%d.\n",
                   rdata_a, rdata_b, ddr_idx_from, i, ddr_idx_to, i);
        }
        status = status || (rdata_a != rdata_b);
    }
    if (status)
    {
        printf("\n\nERROR: Tables are not equal after DMA.\n");
        axi4lpp_del(pp);
        free(pp);
        return 1;
    }

    if (verbose)
        printf("\n\nSuccessfully ran CDMA test.\n");

    if (verbose)
            printf("\r\n--- Exiting main() ---\r\n");
    axi4lpp_del(pp);
    free(pp);
    return 0;
}

int main(int argc, char* argv[])
{
    int verbose = (uint32_t)strtol(argv[1], NULL, 10);
    int status = 0;

    status |= test_cdma(verbose, 0, 1, 0);
    printf((status == 0) ? "PASS" : "FAIL"); printf(": test_cdma 0,1,0\n");
    status |= test_cdma(verbose, 2, 3, 1);
    printf((status == 0) ? "PASS" : "FAIL"); printf(": test_cdma 2,3,1\n");

    // report result
    if(status == 0)
    {
        printf("-- test_cmda: All tests passing --\n");
    }
    else
    {
        printf("-- test_cdma: FAILED --\n");
    }
    return status;
}
