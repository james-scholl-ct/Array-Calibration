
#define MAP_SIZE 4096UL
#define ORCHID_AXI4_ADDR_BASE   0x43C00000

#define ORCHID_AXI4_DWELL_REG           62
#define ORCHID_AXI4_CONTROL_REG         63

#define ORCHID_AXI4_CONTROL_IDX_RESET   0
#define ORCHID_AXI4_CONTROL_IDX_START   1
#define ORCHID_AXI4_CONTROL_IDX_MODE    2

int setup_map(uint32_t **mapped_base_addr, int *dev_mem_fd);
void close_map(uint32_t **mapped_base_addr, int *dev_mem_fd);
void axi_write(uint32_t *base_addr, uint32_t word_addr, uint32_t data);
uint32_t axi_read(uint32_t *base_addr, uint32_t word_addr);

int parse_coeffs_file(const char *filename, uint32_t *array);
uint32_t set_dwell_time_us(uint32_t *base, float dwell_time_us);
void send_reset(uint32_t *base);
void send_start(uint32_t *base, uint32_t mode);
void send_stop(uint32_t *base);
