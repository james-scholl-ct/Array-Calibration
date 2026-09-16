#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "orchid_driver.h"

int main(int argc, char* argv[])
{
    FILE *fp_cmd;
    FILE *fp_rsp;
    FILE *fp_log;

    const char *logfile = "/root/orchid/orchid_app.log";

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
    int dev_mem_fd;
    uint32_t *base;
    fprintf(fp_log, "Setting up memory map.\n");
    setup_map(&base, &dev_mem_fd);
    fprintf(fp_log, "Setup memory map.\n");


    // parse cmd file, execute, and write rsp file
    int nonce;
    char buf[80];
    char cmd[80];
    int n_vars;
    uint32_t addr;
    uint32_t data;

    while(1)
    {
        if( fgets(buf, 80, fp_cmd) != 0 )
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
            else if( strcmp(cmd, "read_bsc") == 0 )
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
                data = axi_read(base, addr);
                fprintf(fp_rsp, "%d read_bsc 0x%x 0x%x\n", nonce, addr, data);
                fprintf(fp_log, "Read 0x%x from bsc address 0x%x.\n", data, addr);
            }
            else if( strcmp(cmd, "write_bsc") == 0 )
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
                axi_write(base, addr, data);
                fprintf(fp_rsp, "%d write_bsc 0x%x 0x%x\n", nonce, addr, data);
                fprintf(fp_log, "Wrote 0x%x to bsc address 0x%x.\n", data, addr);
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

    close_map(&base, &dev_mem_fd);
    fclose(fp_cmd);
    fclose(fp_rsp);
    fclose(fp_log);
    return 0;
}
