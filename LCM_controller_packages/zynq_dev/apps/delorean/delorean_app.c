#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

#include "delorean_driver.h"

#define PORT 1844

#define PAYLOAD_SIZE_MAX 2080

#define DEBUG 0

struct sock_cmd{
    uint8_t code;
    uint16_t length;
    uint8_t payload[PAYLOAD_SIZE_MAX];
};


struct sock_rsp{
    uint8_t code;
    uint16_t length;
    uint8_t payload[PAYLOAD_SIZE_MAX];
};


void dump_struct_sock_cmd(struct sock_cmd *scmd)
{
    fprintf(stderr, "dump_struct_sock_cmd:\n");
    fprintf(stderr, "    code    = %d (0x%02x)\n", scmd->code, scmd->code);
    fprintf(stderr, "    length  = %d (0x%04x)\n", scmd->length, scmd->length);
    if (scmd->length != 0)
    {
        fprintf(stderr, "    payload = \n");
    }
    int i;
    for (i = 0; i < scmd->length; i += 1)
    {
        fprintf(stderr, "        %d: %d (0x%02x)\n", i,
                                                     scmd->payload[i],
                                                     scmd->payload[i]);
    }
}


void dump_struct_sock_rsp(struct sock_rsp *srsp)
{
    fprintf(stderr, "dump_struct_sock_rsp:\n");
    fprintf(stderr, "    code    = %d (0x%02x)\n", srsp->code, srsp->code);
    fprintf(stderr, "    length  = %d (0x%04x)\n", srsp->length, srsp->length);
    if (srsp->length != 0)
    {
        fprintf(stderr, "    payload = \n");
    }
    int i;
    for (i = 0; i < srsp->length; i += 1)
    {
        fprintf(stderr, "        %d: %d (0x%02x)\n", i,
                                                     srsp->payload[i],
                                                     srsp->payload[i]);
    }
}


void wait_us(int us)
{
    // OS can't reliably support sleep < 100 us
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = us * 1000;
    nanosleep(&ts, NULL);
}


int write_sock_sized(int *remote_fd,
                     void *buf_start,
                     uint16_t size,
                     const char *msg)
{
    uint8_t *buf_ptr;
    uint16_t bytes_remaining;
    int n_bytes;

    buf_ptr = (uint8_t *)buf_start;
    bytes_remaining = size;
    while (bytes_remaining > 0)
    {
        n_bytes = write(*remote_fd, buf_ptr, bytes_remaining);
        if (n_bytes == -1)
        {
            perror("ERROR: write_sock_sized failed.\n");
            fprintf(stderr, "ERROR: write_sock_sized failed for '%s' with %d "
                            "bytes remaining of %d.\n", msg, bytes_remaining,
                            size);
            return FAIL;
        }
        buf_ptr += n_bytes;
        bytes_remaining -= n_bytes;
        if (DEBUG)
        {
            fprintf(stderr, "DEBUG: write_sock_sized for '%s' wrote %d "
                            "bytes.\n", msg, n_bytes);
        }
    }
    if (DEBUG)
    {
        fprintf(stderr, "DEBUG: write_sock_sized wrote all %d bytes for "
                        "'%s'.\n", size, msg);
    }

    return SUCCESS;
}


int write_sock(int *remote_fd, struct sock_rsp *srsp)
{
    int status;

    // code
    status = write_sock_sized(remote_fd,
                              &srsp->code,
                              sizeof(srsp->code),
                              "code");
    if (status == FAIL)
    {
        return FAIL;
    }
    if (DEBUG)
    {
        fprintf(stderr, "DEBUG: write_sock sent code 0x%x.\n", srsp->code);
    }


    // length
    status = write_sock_sized(remote_fd,
                              &srsp->length,
                              sizeof(srsp->length),
                              "length");
    if (status == FAIL)
    {
        return FAIL;
    }
    if (DEBUG)
    {
        fprintf(stderr, "DEBUG: write_sock sent length %d.\n", srsp->length);
    }

    // payload
    status = write_sock_sized(remote_fd,
                              srsp->payload,
                              srsp->length,
                              "payload");
    if (status == FAIL)
    {
        return FAIL;
    }

    return SUCCESS;
}


int exec_cmd(struct axi4lpp *pp, struct sock_cmd *scmd, struct sock_rsp *srsp)
{
    int status;
    uint32_t addr;
    uint32_t data;
    switch (scmd->code)
    {
        case 0x00:  // EXIT
            srsp->code = 0x00;
            srsp->length = 0;
            break;

        case 0x01:  // PING
            srsp->code = 0x01;
            srsp->length = 0;
            break;

        case 0x02:  // HOLA
            srsp->code = 0x02;
            srsp->length = 4;
            srsp->payload[0] = 0x48;
            srsp->payload[1] = 0x4f;
            srsp->payload[2] = 0x4c;
            srsp->payload[3] = 0x41;
            break;

        case 0x03:  // ECHO
            srsp->code = 0x03;
            srsp->length = scmd->length;
            memcpy(srsp->payload, scmd->payload, scmd->length);
            break;

        case 0x04:  // READ_LCM
            addr = *( ((uint32_t *)scmd->payload) + 0);
            data = axi4lpp_axi_read(pp->base_lcm, addr);
            srsp->code = 0x04;
            srsp->length = 4;
            memcpy(srsp->payload, &data, 4);
            if (DEBUG)
            {
                fprintf(stderr, "DEBUG: LCM read addr=%d data=0x%08x.\n",
                                addr, data);
            }
            break;

        case 0x05:  // READ_SPI
            addr = *( ((uint32_t *)scmd->payload) + 0);
            data = axi4lpp_axi_read(pp->base_spi, addr);
            srsp->code = 0x05;
            srsp->length = 4;
            memcpy(srsp->payload, &data, 4);
            if (DEBUG)
            {
                fprintf(stderr, "DEBUG: SPI read addr=%d data=0x%08x.\n",
                                addr, data);
            }
            break;

        case 0x06:  // WRITE_LCM
            addr = *( ((uint32_t *)scmd->payload) + 0);
            data = *( ((uint32_t *)scmd->payload) + 1);
            axi4lpp_axi_write(pp->base_lcm, addr, data);
            srsp->code = 0x06;
            srsp->length = 0;
            if (DEBUG)
            {
                fprintf(stderr, "DEBUG: LCM write addr=%d data=0x%08x.\n",
                                addr, data);
            }
            break;

        case 0x07:  // WRITE_SPI
            addr = *( ((uint32_t *)scmd->payload) + 0);
            data = *( ((uint32_t *)scmd->payload) + 1);
            axi4lpp_axi_write(pp->base_spi, addr, data);
            srsp->code = 0x07;
            srsp->length = 0;
            if (DEBUG)
            {
                fprintf(stderr, "DEBUG: SPI write addr=%d data=0x%08x.\n",
                                addr, data);
            }
            break;

        case 0x08:  // CDMA_DOWN
            addr = *( ((uint32_t *)scmd->payload) + 0); // ddr_idx
            data = *( ((uint32_t *)scmd->payload) + 1); // bram_idx
            status = cdma_transfer_down(pp, addr, data);
            srsp->code = 0x08 | (status == SUCCESS ? 0x00 : 0x80);
            srsp->length = 0;
            if (DEBUG)
            {
                fprintf(stderr, "DEBUG: CDMA_DOWN ddr_idx=%d bram_idx=%d "
                                "with SUCCESS=%d.\n",
                                addr, data, (status == SUCCESS));
            }
            break;

        case 0x09:  // CDMA_UP
            addr = *( ((uint32_t *)scmd->payload) + 0); // ddr_idx
            data = *( ((uint32_t *)scmd->payload) + 1); // bram_idx
            status = cdma_transfer_up(pp, addr, data);
            srsp->code = 0x09 | (status == SUCCESS ? 0x00 : 0x80);
            srsp->length = 0;
            if (DEBUG)
            {
                fprintf(stderr, "DEBUG: CDMA_UP ddr_idx=%d bram_idx=%d "
                                "with SUCCESS=%d.\n",
                                addr, data, (status == SUCCESS));
            }
            break;

        case 0x0a:  // CDMA_RESET
            status = cdma_reset_and_wait(pp, 10);
            srsp->code = 0x0a | (status == SUCCESS ? 0x00 : 0x80);
            srsp->length = 0;
            if (DEBUG)
            {
                fprintf(stderr, "DEBUG: CDMA_RESET SUCCESS=%d.\n",
                        (status == SUCCESS));
            }
            break;

        case 0x0b:  // GET_TABLE
            addr = *( ((uint32_t *)scmd->payload) + 0);         // idx
            data = addr * (DMA_BTT_MAX / DATAPATH_BYTESIZE);    // word offset
            srsp->code = 0x0b;
            srsp->length = DMA_BTT_MAX;
            memcpy(srsp->payload, pp->base_ddr + data, DMA_BTT_MAX);
            if (DEBUG)
            {
                fprintf(stderr, "DEBUG: GET_TABLE idx=%d word_offset=%d.\n",
                                addr, data);
            }
            break;

        case 0x0c:  // SET_TABLE
            addr = *( ((uint32_t *)scmd->payload) + 0);         // mask
            data = *( ((uint32_t *)scmd->payload) + 1);         // idx
            uint32_t i;
            uint8_t *ptr_src = scmd->payload + 8;
            uint8_t *ptr_dst = ((uint8_t *)(pp->base_ddr)) + data * DMA_BTT_MAX;
            for (i = 0; i < DMA_BTT_MAX / DATAPATH_BYTESIZE; i += 1)
            {
                if (addr == 3)
                {
                    memcpy(ptr_dst, ptr_src, 4);
                    ptr_src += 4;
                }
                else if (addr == 2)
                {
                    memcpy(ptr_dst + 2, ptr_src, 2);
                    ptr_src += 2;
                }
                else if (addr == 1)
                {
                    memcpy(ptr_dst, ptr_src, 2);
                    ptr_src += 2;
                }
                ptr_dst += DATAPATH_BYTESIZE;
            }
            srsp->code = 0x0c;
            srsp->length = 0;
            if (DEBUG)
            {
                i = data * DMA_BTT_MAX / DATAPATH_BYTESIZE;
                fprintf(stderr, "DEBUG: SET_TABLE mask=%d idx=%d "
                                "word_offset=%d.\n",
                                addr, data, i);
            }
            break;

        case 0x7e:  // SET_FABRIC_RESET
            data = *( ((uint32_t *)scmd->payload) + 0); // reset_en
            status = axi4lpp_set_fabric_reset(pp, data);
            srsp->code = 0x7e | (status == SUCCESS ? 0x00 : 0x80);
            srsp->length = 0;
            break;

        case 0x7f:  // SHUTDOWN
            axi4lpp_shutdown(pp);
            srsp->code = 0x7f;
            srsp->length = 0;
            break;

        default:    // error
            srsp->code = scmd->code | 0x80;
            srsp->length = 0;
            fprintf(stderr, "ERROR: unknown command code %d.\n", scmd->code);
    }
    return SUCCESS;
}


int read_sock_sized(int *remote_fd,
                    void *buf_start,
                    uint16_t size,
                    const char *msg)
{
    uint8_t *buf_ptr;
    uint16_t bytes_remaining;
    int n_bytes;

    buf_ptr = (uint8_t *)buf_start;
    bytes_remaining = size;
    while (bytes_remaining > 0)
    {
        // read blocks until data is received or client hangs up
        n_bytes = read(*remote_fd, buf_ptr, bytes_remaining);
        if (n_bytes < 0)
        {
            perror("ERROR: read_sock_sized failed.\n");
            fprintf(stderr, "ERROR: read_sock_sized failed for '%s' with %d "
                            "bytes remaining of %d.\n", msg, bytes_remaining,
                            size);
            return FAIL;
        }
        else if (n_bytes == 0)
        {
            fprintf(stderr, "ERROR: client closed connection during "
                            "read_sock_sized for '%s' with %d bytes remaining "
                            "of %d.\n", msg, bytes_remaining, size);
            return FAIL;
        }
        buf_ptr += n_bytes;
        bytes_remaining -= n_bytes;
        if (DEBUG)
        {
            fprintf(stderr, "DEBUG: read_sock_sized for '%s' read %d bytes.\n",
                            msg, n_bytes);
        }
    }
    if (DEBUG)
    {
        fprintf(stderr, "DEBUG: read_sock_sized read all %d bytes for "
                        "'%s'.\n", size, msg);
    }

    return SUCCESS;
}


int read_sock(int *remote_fd, struct sock_cmd *scmd)
{
    int status;

    // code
    status = read_sock_sized(remote_fd,
                             &scmd->code,
                             sizeof(scmd->code),
                             "code");
    if (status == FAIL)
    {
        return FAIL;
    }
    if (DEBUG)
    {
        fprintf(stderr, "DEBUG: read_sock read code 0x%x.\n", scmd->code);
    }

    // length
    status = read_sock_sized(remote_fd,
                             &scmd->length,
                             sizeof(scmd->length),
                             "length");
    if (status == FAIL)
    {
        return FAIL;
    }
    if (scmd->length > PAYLOAD_SIZE_MAX)
    {
        fprintf(stderr, "ERROR: length overflowed (%d) for cmd code 0x%x.\n",
                        scmd->length, scmd->code);
        return FAIL;
    }
    if (DEBUG)
    {
        fprintf(stderr, "DEBUG: read_sock read length %d.\n", scmd->length);
    }

    // payload
    status = read_sock_sized(remote_fd,
                             scmd->payload,
                             scmd->length,
                             "payload");
    if (status == FAIL)
    {
        return FAIL;
    }

    return SUCCESS;
}


int accept_new_sock(int *listener_fd, int *remote_fd)
{
    fd_set readfds;
    struct timeval tv;

    FD_ZERO(&readfds);
    FD_SET(*listener_fd, &readfds);

    tv.tv_sec = 15;
    tv.tv_usec = 0;

    // Check listener_fd readiness and update readfds, blocking until timeout
    if (select(*listener_fd + 1, &readfds, NULL, NULL, &tv) == -1)
    {
        perror("ERROR: accept_new_sock failed at select().\n");
        return FAIL;
    }

    if (FD_ISSET(*listener_fd, &readfds))
    {
        // New connection available
        *remote_fd = accept(*listener_fd, 0, 0);
        if (*remote_fd < 0)
        {
            perror("ERROR: accept_new_sock failed at accept().\n");
            return FAIL;
        }

        int opt = 1;
        if (setsockopt(*remote_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt)) < 0)
        {
            close(*remote_fd);
            perror("ERROR: accept_new_sock failed at setsockopt().\n");
            return FAIL;
        }
        return SUCCESS;
    }
    else
    {
        // No connection available after timeout
        fprintf(stderr, "ERROR: accept_new_sock didn't connect before timeout.\n");
        return FAIL;
    }
}


int init_listener_socket(int *listener_fd)
{
    // To accept connections, the following steps are first performed:
    //      1.  A socket is created with socket().
    //      2.  The socket is bound to a local address using bind(), so that
    //          other sockets may be connected to it via connect().
    //      3.  A willingness to accept incoming connections and a queue
    //          limit for incoming connections are specified with listen().

    // PF_INET is synonymous to AF_INET and means IPv4. SOCK_STREAM provides
    // sequenced, reliable, two-way, connection-based byte streams.
    *listener_fd = socket(PF_INET, SOCK_STREAM, 0);
    if (*listener_fd <= 0)
    {
        perror("ERROR: init_listener_socket failed at socket().\n");
        return FAIL;
    }

    // Allow (force) IP and port value reuse from an earlier session, which is
    // otherwise forbidden in order to guarantee there are no stale packets.
    int opt = 1;
    if (setsockopt(*listener_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0)
    {
        close(*listener_fd);
        perror("ERROR: init_listener_socket failed at setsockopt().\n");
        return FAIL;
    }

    // Allow connections from 1) any 2) IPv4 client on 3) given port.
    struct sockaddr_in sai;
    sai.sin_family = AF_INET;
    sai.sin_addr.s_addr = htons(INADDR_ANY);
    sai.sin_port = htons(PORT);

    if (bind(*listener_fd, (struct sockaddr *) &sai, sizeof(sai)) < 0)
    {
        close(*listener_fd);
        perror("ERROR: init_listener_socket failed at bind().\n");
        return FAIL;
    }

    // Backlog (queue depth) is only loosely enforced. Use a common default.
    if (listen(*listener_fd, 128) < 0)
    {
        close(*listener_fd);
        perror("ERROR: init_listener_socket failed at listen().\n");
        return FAIL;
    }

    return SUCCESS;
}


int main(int argc, char* argv[])
{
    int status;
    FILE *fp_log;
    const char *logfile = "/root/delorean/delorean_app.log";
    struct axi4lpp *pp;
    int listener_fd;
    int remote_fd;
    struct sock_cmd scmd;
    struct sock_rsp srsp;

    // Open log file with buffering disabled
    fp_log = fopen(logfile, "w");
    if (!fp_log)
    {
        fprintf(stderr, "ERROR: can't open file %s.\n", logfile);
        exit(-1);
    }
    setbuf(fp_log, 0);
    fprintf(fp_log, "Opened file %s.\n", "<this file>");

    // Parse args
    if(argc != 2)
    {
        fprintf(stderr, "ERROR: you must specify the path of the pattern file "
                        "on the command line.\n");
        fclose(fp_log);
        exit(-1);
    }

    // Init AXI4-Lite peek-poker
    pp = (struct axi4lpp *)malloc(sizeof(struct axi4lpp));
    status = axi4lpp_init(pp);
    if (status == FAIL)
    {
        fprintf(stderr, "ERROR: axi4lpp_init failed.\n");
        axi4lpp_dump(pp, stderr);
        free(pp);
        fclose(fp_log);
        exit(-1);
    }
    fprintf(fp_log, "Created axi4lpp object.\n");

    // Reset CDMA
    status = cdma_reset_and_wait(pp, 10);
    if (status == FAIL)
    {
        fprintf(stderr, "ERROR: cdma_reset_and_wait failed.\n");
        axi4lpp_shutdown(pp);
        axi4lpp_del(pp);
        free(pp);
        fclose(fp_log);
        exit(-1);
    }
    fprintf(fp_log, "CDMA peripheral reset.\n");

    // Init DDR space with pattern memory
    status = axi4lpp_read_pattern_file(pp, argv[1]);
    if (status == FAIL)
    {
        fprintf(stderr, "ERROR: axi4lpp_read_pattern_file failed.\n");
        axi4lpp_shutdown(pp);
        axi4lpp_del(pp);
        free(pp);
        fclose(fp_log);
        exit(-1);
    }
    fprintf(fp_log, "Initialized DDR with patterns from file %s.\n", argv[1]);

    // Setup listener socket
    status = init_listener_socket(&listener_fd);
    if (status == FAIL)
    {
        axi4lpp_shutdown(pp);
        axi4lpp_del(pp);
        free(pp);
        fclose(fp_log);
        exit(-1);
    }
    fprintf(fp_log, "Listening for socket connections.\n");

    // Accept new socket connection (only a single connection supported)
    status = accept_new_sock(&listener_fd, &remote_fd);
    if (status == FAIL)
    {
        close(listener_fd);
        axi4lpp_shutdown(pp);
        axi4lpp_del(pp);
        free(pp);
        fclose(fp_log);
        exit(-1);
    }
    fprintf(fp_log, "Connection accepted.\n");

    // Handle messages from client
    while (1)
    {
        status = read_sock(&remote_fd, &scmd);
        if (status == FAIL)
        {
            fprintf(stderr, "ERROR: Exiting.\n");
            dump_struct_sock_cmd(&scmd);
            dump_struct_sock_rsp(&srsp);
            close(remote_fd);
            close(listener_fd);
            axi4lpp_shutdown(pp);
            axi4lpp_del(pp);
            free(pp);
            fclose(fp_log);
            exit(-1);
        }

        status = exec_cmd(pp, &scmd, &srsp);
        if (status == FAIL)
        {
            fprintf(stderr, "ERROR: Exiting.\n");
            dump_struct_sock_cmd(&scmd);
            dump_struct_sock_rsp(&srsp);
            close(remote_fd);
            close(listener_fd);
            axi4lpp_shutdown(pp);
            axi4lpp_del(pp);
            free(pp);
            fclose(fp_log);
            exit(-1);
        }

        status = write_sock(&remote_fd, &srsp);
        if (status == FAIL)
        {
            fprintf(stderr, "ERROR: Exiting.\n");
            dump_struct_sock_cmd(&scmd);
            dump_struct_sock_rsp(&srsp);
            close(remote_fd);
            close(listener_fd);
            axi4lpp_shutdown(pp);
            axi4lpp_del(pp);
            free(pp);
            fclose(fp_log);
            exit(-1);
        }

        // Special EXIT command
        if (scmd.code == 0x00)
        {
            break;
        }
    }

    // Exit cleanly
    fprintf(fp_log, "Exiting cleanly.\n");
    close(remote_fd);
    close(listener_fd);
    axi4lpp_shutdown(pp);
    axi4lpp_del(pp);
    free(pp);
    fclose(fp_log);
    exit(0);
}
