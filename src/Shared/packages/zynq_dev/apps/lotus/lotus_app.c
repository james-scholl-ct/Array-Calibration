#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

#include "lotus_driver.h"
#include "lotus_driver_bsc.h"
#include "lotus_driver_spi.h"


void wait_us(int us)
{
    // OS can't reliably support sleep < 100 us
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = us * 1000;
    nanosleep(&ts, NULL);
}


void ground_all_channels(struct axi4lpp *pp)
{
    uint32_t vt_words[7] = {0};
    uint32_t gnd_words[7];

    for (int i = 0; i < 7; i = i + 1)
    {
        gnd_words[i] = 0xffffffff;
    }

    axi4lpp_spi_write_jumbo_frame_data(pp, gnd_words, vt_words);
    axi4lpp_spi_send_cmd(pp, JUMBO, DAISY, 12);
}


void select_channel(struct axi4lpp *pp, int vt_channel, int gnd_channel)
{
    int word_addr;
    int bit;
    uint32_t vt_words[7] = {0};
    uint32_t gnd_words[7] = {0};

    word_addr = LOTUS_SPI_SWITCH_MAP[vt_channel][0];
    bit = LOTUS_SPI_SWITCH_MAP[vt_channel][1];
    vt_words[word_addr] = (1 << bit);

    word_addr = LOTUS_SPI_SWITCH_MAP[gnd_channel][0];
    bit = LOTUS_SPI_SWITCH_MAP[gnd_channel][1];
    gnd_words[word_addr] = (1 << bit);

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


void drain_rsp_fifo(struct axi4lpp *pp, uint32_t rsps[64])
{
    uint32_t rsp;
    for (int i = 0; i < 64; i = i + 1)
    {
        rsp = axi4lpp_spi_read_rsp(pp);
        rsps[i] = axi4lpp_spi_get_rsp_is_valid(pp, rsp) ? rsp : 0;
    }
}


void exec_adc_time_series(struct axi4lpp *pp,
                          int meas_channel,
                          int gnd_channel,
                          uint32_t n_wait_cycles,
                          uint32_t rsps[64])
{
    uint32_t n_frames = 63;

    if (! axi4lpp_bsc_driver_is_done(pp))
    {
        return;
    }
    drain_rsp_fifo(pp, rsps);
    reset_switches(pp);
    enable_daisy_chain_mode(pp);

    // switch to both polarities, measuring on first, then ground all
    axi4lpp_spi_set_config(pp, 0x5);
    select_channel(pp, meas_channel, gnd_channel);
    axi4lpp_spi_send_cmd(pp, STREAM, ADC1, n_wait_cycles << 6 | n_frames);
    axi4lpp_spi_set_config(pp, 0x4);
    while (axi4lpp_spi_get_cmd_fifo_count(pp) > 0) {}
    wait_us(100);

    axi4lpp_spi_set_config(pp, 0x5);
    select_channel(pp, gnd_channel, meas_channel);
    axi4lpp_spi_send_cmd(pp, STREAM, ADC1, n_wait_cycles << 6 | n_frames);
    axi4lpp_spi_set_config(pp, 0x4);
    while (axi4lpp_spi_get_cmd_fifo_count(pp) > 0) {}
    wait_us(100);

    ground_all_channels(pp);
    drain_rsp_fifo(pp, rsps);
    axi4lpp_spi_set_config(pp, 0x0);
}

void *ds_handler(void *pp_in)
{
    pthread_detach(pthread_self());
    int ret;
    struct axi4lpp *pp = (struct axi4lpp *)pp_in;
    
    fprintf(stderr, "pp=%#06x\n", (uint32_t)pp);
    
    int listener_fd;
    listener_fd = socket(PF_INET, SOCK_STREAM, 0);
    
    int flag = 1;
    if (setsockopt(listener_fd, SOL_SOCKET, SO_REUSEADDR, &flag, sizeof(flag)) < 0)
    {
        perror("Direct Socket (Setsockopt)");
        return 0;
    }
    
    struct sockaddr_in listener_addr;
    listener_addr.sin_family = AF_INET;
    listener_addr.sin_port = htons(1844);
    listener_addr.sin_addr.s_addr = htons(INADDR_ANY);
    
    if(bind(listener_fd, (struct sockaddr *) &listener_addr, sizeof(listener_addr)) < 0)
    {
        perror("Direct Socket (Bind)");
        return 0;
    }

    if(listen(listener_fd, 128) < 0)
    {
        perror("Direct Socket (Listen)");
        return 0;
    }
    
    int remote_fd = -1;
    
    while (1) {
        if(remote_fd >= 0)
        {
            close(remote_fd);
        }
        
        remote_fd = accept(listener_fd, 0, 0);
        if(remote_fd < 0) {
            perror("Direct Socket (Accept)");
        }
        
        int flag = 1;
        setsockopt(remote_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
    
        while(1) {
            // Grab command byte
            uint8_t command = 0;
            ret = read(remote_fd, &command, sizeof(command));
            if (ret <= 0)
                break;
        
            // For now, only command is 0x02: Set table and apply
            if (command == 0x02) {
                // Get 204 bytes of table
                uint8_t table[204];
                int bytes_remaining = sizeof(table);
                uint8_t *table_fill = table;
            
                while(bytes_remaining > 0) {
                    ret = read(remote_fd, table_fill, bytes_remaining);
                    if (ret <= 0)
                        break;
                    table_fill += ret;
                    bytes_remaining -= ret;
                }
                if(bytes_remaining > 0)
                    break;
            
                // Set and apply table
                for(int i=0; i<51; i++) {
                    axi4lpp_bsc_write(pp, i, *(((uint32_t *)table)+i));
                    // fprintf(stderr, "Writing addr=%#06x data=%#06x\n", i, *(((uint32_t *)table)+i));
                }
                axi4lpp_bsc_write(pp, 0x3e, 0x0c);
            }
            else if (command == 0x04) {
                // Get 1 bytes of table
                uint8_t table[1];
                int bytes_remaining = sizeof(table);
                uint8_t *table_fill = table;
                                
                while(bytes_remaining > 0) {
                    ret = read(remote_fd, table_fill, bytes_remaining);
                    if (ret <= 0)
                        break;
                    table_fill += ret;
                    bytes_remaining -= ret;
                }
                if(bytes_remaining > 0)
                    break;
                uint32_t table_index = (uint32_t)table[0];
                // Load table
                axi4lpp_bsc_write(pp, 60, table_index);
                while(1) {
                    uint32_t rdata;
                    rdata = axi4lpp_bsc_read(pp, 63);
                    if((rdata & 0x1) == 0) {
                            break;
                    }
                }
                // Apply table and enable
                axi4lpp_bsc_write(pp, 0x3e, 0x0c);
            }
            else {
                fprintf(stderr, "Direct Socket: Unknown Command, Dropping Connection\n");
                break;
            }
        
            // Respond with ACK
            uint8_t response = command + 0x01;
            ret = send(remote_fd, &response, sizeof(response), 0);
            if (ret <= 0)
                break;
        }
    }
}


int main(int argc, char* argv[])
{
    FILE *fp_cmd;
    FILE *fp_rsp;
    FILE *fp_log;

    const char *logfile = "/root/lotus/lotus_app.log";

    // open log file with buffering disabled
    fp_log = fopen(logfile, "w");
    if( !fp_log )
    {
        printf("ERROR: can't open file %s.\n", logfile);
        exit(-1);
    }
    setbuf(fp_log, 0);
    fprintf(fp_log, "opened file %s.\n", "<this file>");

    // parse args
    if(argc != 3)
    {
        char *msg = "You must specify the path of the cmd file and the"
                    "path of the rsp file on the command line.\n";
        printf(msg);
        fprintf(fp_log, msg);
        exit(-1);
    }

    // open input cmd file
    fp_cmd = fopen(argv[1], "r");
    if( !fp_cmd )
    {
        char msg[256];
        sprintf(msg, "ERROR: can't open file %s.\n", argv[1]);
        printf(msg);
        fprintf(fp_log, msg);
        exit(-1);
    }
    fprintf(fp_log, "opened file %s.\n", argv[1]);

    // open output rsp file with buffering disabled
    fp_rsp = fopen(argv[2], "w");
    if( !fp_rsp )
    {
        char msg[256];
        sprintf(msg, "ERROR: can't open file %s.\n", argv[2]);
        printf(msg);
        fprintf(fp_log, msg);
        exit(-1);
    }
    setbuf(fp_rsp, 0);
    fprintf(fp_log, "opened file %s.\n", argv[2]);


    // setup memory map for AXI slave iterface
    fprintf(fp_log, "Creating axi4lpp object.\n");
    struct axi4lpp *pp = (struct axi4lpp *)malloc(sizeof(struct axi4lpp));
    axi4lpp_init(pp);
    fprintf(fp_log, "mapped BSC to 0x%x.\n", (unsigned int)pp->base_bsc);
    fprintf(fp_log, "mapped SPI to 0x%x.\n", (unsigned int)pp->base_spi);
    
    // Spin out direct socket handling thread
    pthread_t ds_tid;
    ds_tid = pthread_create(&ds_tid, NULL, ds_handler, (void *) pp);


    // parse cmd file, execute, and write rsp file
    int nonce;
    char buf[80];
    char *bufptr;
    int rlen;
    char cmd[80];
    int n_vars;
    uint32_t addr;
    uint32_t data;
    uint32_t extra;

    while(1)
    {
        // get full (\n terminated) string from the command file
        bufptr = buf;
        while(1)
        {
            rlen = fread(bufptr, 1, 1, fp_cmd);
            if (rlen > 0)
            {
                bufptr += rlen;
                if (*(bufptr - 1) == '\n' || (bufptr - buf) == sizeof(buf) - 1)
                {
                    break;
                }
            }
        }
        *bufptr = '\0';

        if (1)
        {
            // parse cmd type
            n_vars = sscanf(buf, "%d %s", &nonce, cmd);
            if( n_vars != 2 )
            {
                fprintf(fp_log, "Error parsing cmd: %s", buf);
                fprintf(fp_log, "Exiting.\n");
                break;
            }
            else
            {
                fprintf(fp_log, "Scanned command type '%s' from: %s", cmd, buf);
            }

            // execute cmd
            if( strcmp(cmd, "exit") == 0 )
            {
                // exit command
                fprintf(fp_rsp, "%d exit\n", nonce);
                fprintf(fp_log, "Exiting normally.\n");
                break;
            }
            else if( strcmp(cmd, "read_bsc") == 0 || strcmp(cmd, "read_spi") == 0)
            {
                // read command
                n_vars = sscanf(buf, "%d %s 0x%x", &nonce, cmd, &addr);
                if( n_vars != 3 )
                {
                    fprintf(fp_rsp, "Error parsing read: %s", buf);
                    fprintf(fp_log, "Error parsing read: %s", buf);
                    fprintf(fp_log, "Exiting.\n");
                    break;
                }
                if( strcmp(cmd, "read_spi") == 0 )
                {
                    data = axi4lpp_spi_read(pp, addr);
                    fprintf(fp_rsp, "%d read_spi 0x%x 0x%x\n", nonce, addr, data);
                    fprintf(fp_log, "Read 0x%x from spi address 0x%x.\n", data, addr);
                }
                else
                {
                    data = axi4lpp_bsc_read(pp, addr);
                    fprintf(fp_rsp, "%d read_bsc 0x%x 0x%x\n", nonce, addr, data);
                    fprintf(fp_log, "Read 0x%x from bsc address 0x%x.\n", data, addr);
                }
            }
            else if( strcmp(cmd, "write_bsc") == 0 || strcmp(cmd, "write_spi") == 0)
            {
                // write command
                n_vars = sscanf(buf, "%d %s 0x%x 0x%x", &nonce, cmd, &addr, &data);
                if( n_vars != 4 )
                {
                    fprintf(fp_rsp, "Error parsing write: %s", buf);
                    fprintf(fp_log, "Error parsing write: %s", buf);
                    fprintf(fp_log, "Exiting.\n");
                    break;
                }
                if( strcmp(cmd, "write_spi") == 0 )
                {
                    axi4lpp_spi_write(pp, addr, data);
                    fprintf(fp_rsp, "%d write_spi 0x%x 0x%x\n", nonce, addr, data);
                    fprintf(fp_log, "Wrote 0x%x to spi address 0x%x.\n", data, addr);
                }
                else
                {
                    axi4lpp_bsc_write(pp, addr, data);
                    fprintf(fp_rsp, "%d write_bsc 0x%x 0x%x\n", nonce, addr, data);
                    fprintf(fp_log, "Wrote 0x%x to bsc address 0x%x.\n", data, addr);
                }
            }
            else if( strcmp(cmd, "exec_adc_time_series") == 0 )
            {
                // exec_adc_time_series meas_channel gnd_channel n_wait_cycles
                n_vars = sscanf(buf, "%d %s %d %d %d", &nonce, cmd, &addr, &data, &extra);
                if( n_vars != 5 )
                {
                    fprintf(fp_rsp, "Error parsing exec_adc_time_series: %s", buf);
                    fprintf(fp_log, "Error parsing exec_adc_time_series: %s", buf);
                    fprintf(fp_log, "Exiting.\n");
                    break;
                }
                uint32_t rsps[64];
                exec_adc_time_series(pp, addr, data, extra, rsps);
                fprintf(fp_rsp, "%d exec_adc_time_series %d %d %d", nonce, addr, data, extra);
                for (int i = 0; i < 64; i = i + 1)
                {
                    if(rsps[i] != 0)
                    {
                        fprintf(fp_rsp, " 0x%x", rsps[i]);
                    }
                }
                fprintf(fp_rsp, "\n");
                fprintf(fp_log, "Ran exec_adc_time_series for rails %d and %d.\n", addr, data);
            }
            else if( strcmp(cmd, "exec_adc_time_series_full") == 0 )
            {
                // exec_adc_time_series_full n_wait_cycles
                n_vars = sscanf(buf, "%d %s %d", &nonce, cmd, &extra);
                if( n_vars != 3 )
                {
                    fprintf(fp_rsp, "Error parsing exec_adc_time_series_full: %s", buf);
                    fprintf(fp_log, "Error parsing exec_adc_time_series_full: %s", buf);
                    fprintf(fp_log, "Exiting.\n");
                    break;
                }

                const char *csv_file = "/root/lotus/exec_adc_time_series_full.csv";
                FILE *fp_csv = fopen(csv_file, "w");
                if( !fp_csv )
                {
                    char msg[256];
                    sprintf(msg, "ERROR: can't open file %s.\n", csv_file);
                    printf(msg);
                    fprintf(fp_log, msg);
                    exit(-1);
                }
                fprintf(fp_log, "opened file %s.\n", csv_file);

                for (int i = 0; i < 204; i = i + 1)
                {
                    for (int j = 0; j < i; j = j + 1)
                    {
                        uint32_t rsps[64] = {0};
                        exec_adc_time_series(pp, i, j, extra, rsps);
                        fprintf(fp_csv, "%d,%d", i, j);
                        for (int k = 0; k < 64; k = k + 1)
                        {
                            fprintf(fp_csv, ",0x%x", rsps[k]);
                        }
                        fprintf(fp_csv, "\n");
                    }
                }
                fclose(fp_csv);
                fprintf(fp_rsp, "%d exec_adc_time_series_full %d %s\n", nonce, extra, csv_file);
                fprintf(fp_log, "Ran exec_adc_time_series_full with n_wait_cycles = %d.\n", extra);
            }
            else if( strcmp(cmd, "exec_adc_time_series_nearest_neighbors") == 0 )
            {
                // exec_adc_time_series_full n_wait_cycles
                n_vars = sscanf(buf, "%d %s %d", &nonce, cmd, &extra);
                if( n_vars != 3 )
                {
                    fprintf(fp_rsp, "Error parsing exec_adc_time_series_nearest_neighbors: %s", buf);
                    fprintf(fp_log, "Error parsing exec_adc_time_series_nearest_neighbors: %s", buf);
                    fprintf(fp_log, "Exiting.\n");
                    break;
                }

                const char *csv_file = "/root/lotus/exec_adc_time_series_nearest_neighbors.csv";
                FILE *fp_csv = fopen(csv_file, "w");
                if( !fp_csv )
                {
                    char msg[256];
                    sprintf(msg, "ERROR: can't open file %s.\n", csv_file);
                    printf(msg);
                    fprintf(fp_log, msg);
                    exit(-1);
                }
                fprintf(fp_log, "opened file %s.\n", csv_file);

                for (int i = 0; i < 204; i = i + 1)
                {
                    int j;
                    j = (i - 1);
                    if (j == -1)
                    {
                        j = 203;
                    }
                    uint32_t rsps[64] = {0};
                    exec_adc_time_series(pp, i, j, extra, rsps);
                    fprintf(fp_csv, "%d,%d", i, j);
                    for (int k = 0; k < 64; k = k + 1)
                    {
                        fprintf(fp_csv, ",0x%x", rsps[k]);
                    }
                    fprintf(fp_csv, "\n");
                }
                fclose(fp_csv);
                fprintf(fp_rsp, "%d exec_adc_time_series_nearest_neighbors %d %s\n", nonce, extra, csv_file);
                fprintf(fp_log, "Ran exec_adc_time_series_nearest_neighbors with n_wait_cycles = %d.\n", extra);
            }
            else
            {
                // unrecognized command
                fprintf(fp_rsp, "Unrecognized cmd: %s.\n", cmd);
                fprintf(fp_log, "Unrecognized cmd: %s.\n", cmd);
                fprintf(fp_log, "Exiting.\n");
                break;
            }
        }
    }

    fprintf(stderr, "Exiting.\n");
    axi4lpp_del(pp);
    free(pp);
    fclose(fp_cmd);
    fclose(fp_rsp);
    fclose(fp_log);
    return 0;
}
