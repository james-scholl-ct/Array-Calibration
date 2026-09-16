#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "genesis_driver.h"
#include "genesis_driver_ip0.h"
#include "genesis_driver_ip1.h"


void wait_us(int us)
{
    // OS can't reliably support sleep < 100 us
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = us * 1000;
    nanosleep(&ts, NULL);
}


int main(int argc, char* argv[])
{
    FILE *fp_cmd;
    FILE *fp_rsp;
    FILE *fp_log;

    const char *logfile = "/root/genesis/genesis_app.log";

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
    fprintf(fp_log, "mapped IP0 to 0x%x.\n", (unsigned int)pp->base_ip0);
    fprintf(fp_log, "mapped IP1 to 0x%x.\n", (unsigned int)pp->base_ip1);


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
            else if( strcmp(cmd, "read_ip0") == 0 || strcmp(cmd, "read_ip1") == 0)
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
                if( strcmp(cmd, "read_ip1") == 0 )
                {
                    data = axi4lpp_ip1_read(pp, addr);
                    fprintf(fp_rsp, "%d read_ip1 0x%x 0x%x\n", nonce, addr, data);
                    fprintf(fp_log, "Read 0x%x from ip1 address 0x%x.\n", data, addr);
                }
                else
                {
                    data = axi4lpp_ip0_read(pp, addr);
                    fprintf(fp_rsp, "%d read_ip0 0x%x 0x%x\n", nonce, addr, data);
                    fprintf(fp_log, "Read 0x%x from ip0 address 0x%x.\n", data, addr);
                }
            }
            else if( strcmp(cmd, "write_ip0") == 0 || strcmp(cmd, "write_ip1") == 0)
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
                if( strcmp(cmd, "write_ip1") == 0 )
                {
                    axi4lpp_ip1_write(pp, addr, data);
                    fprintf(fp_rsp, "%d write_ip1 0x%x 0x%x\n", nonce, addr, data);
                    fprintf(fp_log, "Wrote 0x%x to ip1 address 0x%x.\n", data, addr);
                }
                else
                {
                    axi4lpp_ip0_write(pp, addr, data);
                    fprintf(fp_rsp, "%d write_ip0 0x%x 0x%x\n", nonce, addr, data);
                    fprintf(fp_log, "Wrote 0x%x to ip0 address 0x%x.\n", data, addr);
                }
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

    axi4lpp_del(pp);
    free(pp);
    fclose(fp_cmd);
    fclose(fp_rsp);
    fclose(fp_log);
    return 0;
}
