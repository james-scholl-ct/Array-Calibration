#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "orchid_driver.h"


int test_axi_rw()
{
    int status = 0;

    int dev_mem_fd;
    uint32_t *base;
    setup_map(&base, &dev_mem_fd);

    // Test reads and writes
    uint32_t i;
    for(i = 0; i < 64; i = i + 1)
    {
        uint32_t data = i ^ 0xdeadbeef;
        axi_write(base, i, data);
    }

    for(i = 0; i < 64; i = i + 1)
    {
        uint32_t read = axi_read(base, i);
        uint32_t expected = i ^ 0xdeadbeef;
        if(read != expected)
        {
            printf("ERROR: read 0x%08x, expecting 0x%08x\n", read, expected);
            status = 1;
        }
    }

    // Test built-in adder
    volatile uint32_t *addend_0_addr = base + 64;
    volatile uint32_t *addend_1_addr = base + 65;
    volatile uint32_t *sum_addr = base + 66;

    uint32_t a = 0xdead00ff;
    uint32_t b = 0x00000001;
    *addend_0_addr = a;
    *addend_1_addr = b;
    uint32_t sum = *sum_addr;
    if (sum != a + b)
    {
        printf("ERROR: sum = 0x%08x\n", sum);
        status = 1;
    }

    close_map(&base, &dev_mem_fd);

    // report result
    if(status == 0)
    {
        printf("pass: test_axi_rw\n");
    }
    else
    {
        printf("FAIL: test_axi_rw\n");
    }
    return status;
}

int test_file_parse()
{
    int status = 0;

    uint32_t array[64];
    const char *filename = "coeffs_dummy.txt";
    parse_coeffs_file(filename, array);

    int i;
    for ( i = 0; i < 64; i = i + 1)
    {
        if(i != array[i])
        {
            printf("ERROR: i = %d, array[i] = %d\n", i, array[i]);
            status = 1;
        }
    }

    // report result
    if(status == 0)
    {
        printf("pass: test_file_parse\n");
    }
    else
    {
        printf("FAIL: test_file_parse\n");
    }
    return status;
}


int test_set_dwell_time_us()
{
    int status = 0;

    int dev_mem_fd;
    uint32_t *base;
    setup_map(&base, &dev_mem_fd);

    uint32_t written_data;
    uint32_t expected;
    written_data = set_dwell_time_us(base, 0.0166);
    expected = 1;
    if (written_data != expected)
    {
        printf("ERROR: wrote %d, expected %d\n", written_data, expected);
        status = 1;
    }

    written_data = set_dwell_time_us(base, 1);
    expected = 60;
    if (written_data != expected)
    {
        printf("ERROR: wrote %d, expected %d\n", written_data, expected);
        status = 1;
    }

    close_map(&base, &dev_mem_fd);

    // report result
    if(status == 0)
    {
        printf("pass: test_set_dwell_time_us\n");
    }
    else
    {
        printf("FAIL: test_set_dwell_time_us\n");
    }
    return status;
}



int main(){
    int status = 0;
    status |= test_axi_rw();
    status |= test_file_parse();
    status |= test_set_dwell_time_us();

    // report result
    if(status == 0)
    {
        printf("-- All tests passing --\n");
    }
    else
    {
        printf("-- Tests FAILED --\n");
    }
    return status;
}
