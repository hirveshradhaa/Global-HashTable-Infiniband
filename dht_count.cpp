#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <regex>
#include <vector>
#include <unordered_map>
#include <mpi.h>
#include <infiniband/verbs.h>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>
#include <errno.h>
#include <cstdlib>
#include <ctime>

#define DEBUG_RDMA 0
#define DEBUG_QP 0

struct WordRecord;
struct BucketHeader;
struct Bucket;
struct WordOccurrence;
struct RDMAContext;
struct RDMAConnection;
struct RDMABuffer;
struct QPInfo;

void debug_log(const char* prefix, const char* msg, int rank) {
    if (DEBUG_RDMA) {
        printf("[Rank %d] %s: %s\n", rank, prefix, msg);
        fflush(stdout);
    }
}

#pragma pack(push, 1)
struct WordRecord {
    char word[64] = {0};
    uint64_t frequency = 0;
    uint64_t locations[7] = {0};
} __attribute__((aligned(8)));  
static_assert(sizeof(WordRecord) == 128, "WordRecord size must be 128 bytes");

struct BucketHeader {
    uint64_t record_count = 0;
    uint8_t reserved[120] = {0};
} __attribute__((aligned(8)));  
static_assert(sizeof(BucketHeader) == 128, "BucketHeader size must be 128 bytes");

struct Bucket {
    BucketHeader header{};
    WordRecord records[32767]{};

    Bucket() = default; 
} __attribute__((aligned(4096)));  
static_assert(sizeof(Bucket) == 4 * 1024 * 1024, "Bucket size must be 4MB");
#pragma pack(pop)

struct WordOccurrence {
    char word[64] = {0};
    uint64_t location = 0;
};

struct RDMAConnection {
    struct ibv_qp* qp = nullptr;
    uint32_t remote_qpn = 0;
    uint16_t remote_lid = 0;
    uint32_t remote_psn = 0;
    uint64_t remote_addr = 0;
    uint32_t remote_rkey = 0;
};

struct RDMAContext {
    struct ibv_context* context = nullptr;
    struct ibv_pd* pd = nullptr;
    struct ibv_mr* mr = nullptr;
    struct ibv_cq* cq = nullptr;
    struct ibv_comp_channel* comp_channel = nullptr;
    RDMAConnection* connections = nullptr;
    int rank = -1;
};

struct QPInfo {
    uint32_t qp_num = 0;
    uint16_t lid = 0;
    uint32_t psn = 0;
    uint64_t addr = 0;
    uint32_t rkey = 0;
};

struct RDMABuffer {
    void* addr = nullptr;
    struct ibv_mr* mr = nullptr;
    size_t size = 0;
};
#pragma pack(pop)


struct ibv_qp* create_qp(RDMAContext* ctx);
unsigned int hash_function(const std::string& word);
std::string to_uppercase(const std::string& word);
std::vector<std::pair<std::string, uint64_t>> extract_words_with_indices(const std::string& text, uint64_t offset);
void init_rdma(RDMAContext* ctx, Bucket* local_buckets);
void setup_rdma_connections(RDMAContext* ctx, int rank, int size, MPI_Comm comm);
void cleanup_rdma(RDMAContext* ctx);
bool post_rdma_read(RDMAContext* ctx, void* local_addr, size_t len, uint64_t remote_addr, uint32_t remote_rkey);
bool wait_for_completion(RDMAContext* ctx);
std::vector<std::string> read_query_file(const std::string& filename);
void process_file(const std::string& filename, Bucket* local_buckets, int rank, int size, MPI_Comm comm);
void query_words(const std::vector<std::string>& query_words, Bucket* local_buckets, int rank, int size, MPI_Comm comm);
void display_results(Bucket* local_buckets, int rank, int size, MPI_Comm comm);
void debug_qp_state(struct ibv_qp* qp, const char* prefix, int rank);
void debug_mr(RDMAContext* ctx, const char* prefix);
void debug_qp_attr(const ibv_qp_attr* attr, const char* prefix, int rank);


void* aligned_alloc_wrapper(size_t alignment, size_t size) {
    void* ptr;
    int ret = posix_memalign(&ptr, alignment, size);
    if (ret != 0) {
        std::cerr << "Failed to allocate aligned memory: " << strerror(ret) << std::endl;
        return nullptr;
    }
    return ptr;
}


void debug_qp_attr(const ibv_qp_attr* attr, const char* prefix, int rank) {
    char debug_msg[512];
    snprintf(debug_msg, sizeof(debug_msg), 
             "%s - QP attributes: state=%d, cur_qp_state=%d, path_mtu=%d, "
             "path_mig_state=%d, qkey=%u, rq_psn=%u, sq_psn=%u, dest_qp_num=%u, "
             "qp_access_flags=%x", 
             prefix, attr->qp_state, attr->cur_qp_state, attr->path_mtu, 
             attr->path_mig_state, attr->qkey, attr->rq_psn, attr->sq_psn,
             attr->dest_qp_num, attr->qp_access_flags);
    debug_log("QP", debug_msg, rank);
}

void debug_mr(RDMAContext* ctx, const char* prefix) {
    char debug_msg[256];
    snprintf(debug_msg, sizeof(debug_msg), 
             "%s - MR: addr=%p, length=%lu, lkey=%u, rkey=%u",
             prefix, ctx->mr->addr, ctx->mr->length, 
             ctx->mr->lkey, ctx->mr->rkey);
    debug_log("MR", debug_msg, ctx->rank);
}

struct ibv_qp* create_qp(RDMAContext* ctx) {
    struct ibv_qp_init_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.send_cq = ctx->cq;
    attr.recv_cq = ctx->cq;
    attr.qp_type = IBV_QPT_RC;
    attr.cap.max_send_wr = 100;
    attr.cap.max_recv_wr = 1;
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;
    attr.sq_sig_all = 0;

    struct ibv_qp* qp = ibv_create_qp(ctx->pd, &attr);
    if (!qp) {
        debug_log("ERROR", "Failed to create QP", ctx->rank);
        return nullptr;
    }

    debug_log("QP", "QP created successfully", ctx->rank);
    return qp;
}

RDMABuffer allocate_rdma_buffer(RDMAContext* ctx, size_t size) {
    RDMABuffer buffer;
    buffer.size = size;
    buffer.addr = aligned_alloc(4096, size); 
    if (!buffer.addr) {
        debug_log("ERROR", "Failed to allocate RDMA buffer", ctx->rank);
        exit(1);
    }

    buffer.mr = ibv_reg_mr(ctx->pd, buffer.addr, size,
                          IBV_ACCESS_LOCAL_WRITE |
                          IBV_ACCESS_REMOTE_READ |
                          IBV_ACCESS_REMOTE_WRITE);
    if (!buffer.mr) {
        debug_log("ERROR", "Failed to register RDMA buffer", ctx->rank);
        free(buffer.addr);
        exit(1);
    }
    
    return buffer;
}

bool check_qp_ready(struct ibv_qp* qp) {
    struct ibv_qp_attr attr;
    struct ibv_qp_init_attr init_attr;
    
    if (ibv_query_qp(qp, &attr, IBV_QP_STATE, &init_attr)) {
        return false;
    }
    
    return attr.qp_state == IBV_QPS_RTS;
}

void init_rdma(RDMAContext* ctx, Bucket* local_buckets, int rank) {
    ctx->rank = rank;
    debug_log("INIT", "Starting RDMA initialization", rank);

    
    ctx->connections = new RDMAConnection[8]();
    
    
    int num_devices;
    struct ibv_device **dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list) {
        debug_log("ERROR", "Failed to get IB devices list", rank);
        exit(1);
    }

    
    struct ibv_device *ib_dev = nullptr;
    for (int i = 0; i < num_devices; ++i) {
        if (strncmp(ibv_get_device_name(dev_list[i]), "mlx5_0", 6) == 0) {
            ib_dev = dev_list[i];
            char debug_msg[100];
            snprintf(debug_msg, sizeof(debug_msg), "Found device: %s", 
                    ibv_get_device_name(dev_list[i]));
            debug_log("INIT", debug_msg, rank);
            break;
        }
    }

    if (!ib_dev) {
        debug_log("ERROR", "IB device mlx5_0 not found", rank);
        ibv_free_device_list(dev_list);
        exit(1);
    }

    
    ctx->context = ibv_open_device(ib_dev);
    if (!ctx->context) {
        debug_log("ERROR", "Failed to open device context", rank);
        ibv_free_device_list(dev_list);
        exit(1);
    }

    
    struct ibv_device_attr device_attr;
    if (ibv_query_device(ctx->context, &device_attr)) {
        debug_log("ERROR", "Failed to query device attributes", rank);
        exit(1);
    }


    ctx->comp_channel = ibv_create_comp_channel(ctx->context);
    if (!ctx->comp_channel) {
        debug_log("ERROR", "Failed to create completion channel", rank);
        exit(1);
    }


    ctx->pd = ibv_alloc_pd(ctx->context);
    if (!ctx->pd) {
        debug_log("ERROR", "Failed to allocate protection domain", rank);
        exit(1);
    }

    size_t total_size = sizeof(Bucket) * 256;
int access_flags = IBV_ACCESS_LOCAL_WRITE | 
                  IBV_ACCESS_REMOTE_READ | 
                  IBV_ACCESS_REMOTE_WRITE |
                  IBV_ACCESS_REMOTE_ATOMIC;  

ctx->mr = ibv_reg_mr(ctx->pd, local_buckets, total_size, access_flags);

    
    if ((uintptr_t)local_buckets % 4096 != 0) {
        debug_log("ERROR", "Memory not properly aligned for RDMA", rank);
        exit(1);
    }

    ctx->mr = ibv_reg_mr(ctx->pd, local_buckets, total_size, access_flags);
    if (!ctx->mr) {
        char error_msg[256];
        snprintf(error_msg, sizeof(error_msg), 
                "Failed to register memory region: %s", strerror(errno));
        debug_log("ERROR", error_msg, rank);
        exit(1);
    }

    
    int cq_size = 100; 
    ctx->cq = ibv_create_cq(ctx->context, cq_size, nullptr, 
                           ctx->comp_channel, 0);
    if (!ctx->cq) {
        debug_log("ERROR", "Failed to create CQ", rank);
        exit(1);
    }

    
    if (ibv_req_notify_cq(ctx->cq, 0)) {
        debug_log("ERROR", "Failed to request CQ notifications", rank);
        exit(1);
    }

    
    struct ibv_port_attr port_attr;
    if (ibv_query_port(ctx->context, 1, &port_attr)) {
        debug_log("ERROR", "Failed to query port attributes", rank);
        exit(1);
    }

    if (port_attr.state != IBV_PORT_ACTIVE) {
        debug_log("ERROR", "IB port is not active", rank);
        exit(1);
    }

    
    char debug_msg[256];
    snprintf(debug_msg, sizeof(debug_msg), 
             "RDMA initialization complete - Device: %s, PD: %p, MR: %p, CQ: %p", 
             ibv_get_device_name(ib_dev), (void*)ctx->pd, (void*)ctx->mr, 
             (void*)ctx->cq);
    debug_log("INIT", debug_msg, rank);

    
    snprintf(debug_msg, sizeof(debug_msg), 
             "Memory Region - addr=%p, length=%zu, lkey=%u, rkey=%u", 
             ctx->mr->addr, ctx->mr->length, ctx->mr->lkey, ctx->mr->rkey);
    debug_log("INIT", debug_msg, rank);

    ibv_free_device_list(dev_list);

    __sync_synchronize();
}


void transition_qp_state(struct ibv_qp *qp, enum ibv_qp_state target_state, int rank) {
    char debug_msg[100];
    snprintf(debug_msg, sizeof(debug_msg), "Transitioning QP to state %d", target_state);
    debug_log("QP", debug_msg, rank);
    
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = target_state;
    
    int flags = IBV_QP_STATE;
    
    switch (target_state) {
        case IBV_QPS_INIT:
            attr.pkey_index = 0;
            attr.port_num = 1;
            attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | 
                                 IBV_ACCESS_REMOTE_READ | 
                                 IBV_ACCESS_REMOTE_WRITE;
            flags |= IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
            break;
            
        case IBV_QPS_RTR:
            break;
            
        case IBV_QPS_RTS:
            break;
            
        default:
            debug_log("ERROR", "Invalid QP state transition requested", rank);
            return;
    }
    
    if (ibv_modify_qp(qp, &attr, flags)) {
        debug_log("ERROR", "Failed to modify QP state", rank);
        return;
    }
    
    debug_log("QP", "QP state transition successful", rank);
}

unsigned int hash_function(const std::string& word) {
    unsigned int hash = 0;
    const int multipliers[2] = {121, 1331}; // 11^2 and 11^3
    for (size_t i = 0; i < word.size(); ++i) {
        int Wi = static_cast<int>(word[i]);
        int multiplier = multipliers[i % 2];
        hash = (hash + Wi * multiplier) % 2048;
    }
    return hash;
}

std::string to_uppercase(const std::string& word) {
    std::string upper_word = word;
    std::transform(upper_word.begin(), upper_word.end(), upper_word.begin(), ::toupper);
    return upper_word;
}

std::vector<std::pair<std::string, uint64_t>> extract_words_with_indices(const std::string& text, uint64_t offset) {
    std::vector<std::pair<std::string, uint64_t>> words_with_indices;
    std::regex word_regex(R"([a-zA-Z0-9]{1,62})");
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), word_regex);
    auto words_end = std::sregex_iterator();

    for (auto it = words_begin; it != words_end; ++it) {
        uint64_t start_index = it->position() + offset;
        words_with_indices.emplace_back(it->str(), start_index);
    }
    return words_with_indices;
}



void query_words(const std::vector<std::string>& query_words, Bucket* local_buckets, int rank, int size, MPI_Comm comm);
void display_results(Bucket* local_buckets, int rank, int size, MPI_Comm comm);
void process_file(const std::string& filename, Bucket* local_buckets, int rank, int size, MPI_Comm comm);
std::vector<std::string> read_query_file(const std::string& filename);

bool check_qp_state(struct ibv_qp* qp) {
    struct ibv_qp_attr attr;
    struct ibv_qp_init_attr init_attr;
    memset(&attr, 0, sizeof(attr));
    memset(&init_attr, 0, sizeof(init_attr));
    
    if (ibv_query_qp(qp, &attr, IBV_QP_STATE, &init_attr)) {
        return false;
    }
    
    return attr.qp_state == IBV_QPS_RTS;
}

void debug_completion(const ibv_wc* wc, const char* prefix, int rank) {
    char debug_msg[512];
    snprintf(debug_msg, sizeof(debug_msg),
             "%s - status=%s(%d), wr_id=%lu, qp_num=%u, vendor_err=0x%x, "
             "byte_len=%u, opcode=%d",
             prefix, ibv_wc_status_str(wc->status), wc->status, wc->wr_id,
             wc->qp_num, wc->vendor_err, wc->byte_len, wc->opcode);
    debug_log("COMP", debug_msg, rank);
}

void debug_qp_state(ibv_qp* qp, const char* prefix, int rank) {
    struct ibv_qp_attr attr;
    struct ibv_qp_init_attr init_attr;
    memset(&attr, 0, sizeof(attr));
    memset(&init_attr, 0, sizeof(init_attr));

    if (ibv_query_qp(qp, &attr, IBV_QP_STATE | IBV_QP_TIMEOUT | 
                     IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | 
                     IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC,
                     &init_attr) == 0) {
        char debug_msg[256];
        const char* state_str;
        switch (attr.qp_state) {
            case IBV_QPS_RESET: state_str = "RESET"; break;
            case IBV_QPS_INIT: state_str = "INIT"; break;
            case IBV_QPS_RTR: state_str = "RTR"; break;
            case IBV_QPS_RTS: state_str = "RTS"; break;
            case IBV_QPS_SQD: state_str = "SQD"; break;
            case IBV_QPS_SQE: state_str = "SQE"; break;
            case IBV_QPS_ERR: state_str = "ERR"; break;
            default: state_str = "UNKNOWN"; break;
        }
        snprintf(debug_msg, sizeof(debug_msg), 
                "%s - QP State: %s(%d), PSN: %u, timeout: %d, retry_cnt: %d",
                prefix, state_str, attr.qp_state, attr.sq_psn, attr.timeout, attr.retry_cnt);
        debug_log("QP", debug_msg, rank);
        
        debug_qp_attr(&attr, "Full QP Attributes", rank);
    } else {
        char debug_msg[256];
        snprintf(debug_msg, sizeof(debug_msg),
                "%s - Failed to query QP state: %s", prefix, strerror(errno));
        debug_log("ERROR", debug_msg, rank);
    }
}

WordRecord rdma_read_record(RDMAContext* ctx, int target_rank, int bucket_number, const std::string& word) {
    char debug_msg[512];
    WordRecord result{};

    if (target_rank >= 8 || !ctx->connections[target_rank].qp) {
        snprintf(debug_msg, sizeof(debug_msg),
                "Invalid connection - target_rank=%d, qp=%p",
                target_rank, (void*)ctx->connections[target_rank].qp);
        debug_log("ERROR", debug_msg, ctx->rank);
        return result;
    }

    struct ibv_qp_attr attr;
    struct ibv_qp_init_attr init_attr;
    if (ibv_query_qp(ctx->connections[target_rank].qp, &attr, IBV_QP_STATE, &init_attr)) {
        debug_log("ERROR", "Failed to query QP state", ctx->rank);
        return result;
    }

    if (attr.qp_state != IBV_QPS_RTS) {
        debug_log("ERROR", "QP not in RTS state", ctx->rank);
        return result;
    }

    size_t bucket_size = sizeof(Bucket);
    uint64_t bucket_offset = static_cast<uint64_t>(bucket_number) * bucket_size;
    uint64_t remote_addr = ctx->connections[target_rank].remote_addr + bucket_offset;

    void* header_buf = aligned_alloc(4096, sizeof(BucketHeader));
    if (!header_buf) {
        debug_log("ERROR", "Failed to allocate aligned buffer for header", ctx->rank);
        return result;
    }

    struct ibv_mr* header_mr = ibv_reg_mr(ctx->pd, header_buf, sizeof(BucketHeader),
                                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!header_mr) {
        debug_log("ERROR", "Failed to register MR for header", ctx->rank);
        free(header_buf);
        return result;
    }

    struct ibv_sge sge{};
    sge.addr = (uint64_t)header_buf;
    sge.length = sizeof(BucketHeader);
    sge.lkey = header_mr->lkey;

    struct ibv_send_wr wr{};
    struct ibv_send_wr* bad_wr = nullptr;
    wr.wr_id = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = ctx->connections[target_rank].remote_rkey;

    if (ibv_post_send(ctx->connections[target_rank].qp, &wr, &bad_wr)) {
        debug_log("ERROR", "Failed to post header read", ctx->rank);
        ibv_dereg_mr(header_mr);
        free(header_buf);
        return result;
    }

    struct ibv_wc wc{};
    int ne;
    do {
        ne = ibv_poll_cq(ctx->cq, 1, &wc);
    } while (ne == 0);

    if (ne < 0 || wc.status != IBV_WC_SUCCESS) {
        snprintf(debug_msg, sizeof(debug_msg),
                "Header read failed with status: %s", ibv_wc_status_str(wc.status));
        debug_log("ERROR", debug_msg, ctx->rank);
        ibv_dereg_mr(header_mr);
        free(header_buf);
        return result;
    }

    BucketHeader* header = static_cast<BucketHeader*>(header_buf);

    if (header->record_count == 0) {
        ibv_dereg_mr(header_mr);
        free(header_buf);
        return result;
    }

    size_t records_size = sizeof(WordRecord) * header->record_count;
    void* records_buf = aligned_alloc(4096, records_size);
    if (!records_buf) {
        debug_log("ERROR", "Failed to allocate aligned buffer for records", ctx->rank);
        ibv_dereg_mr(header_mr);
        free(header_buf);
        return result;
    }

    struct ibv_mr* records_mr = ibv_reg_mr(ctx->pd, records_buf, records_size,
                                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!records_mr) {
        debug_log("ERROR", "Failed to register MR for records", ctx->rank);
        free(records_buf);
        ibv_dereg_mr(header_mr);
        free(header_buf);
        return result;
    }

    sge.addr = (uint64_t)records_buf;
    sge.length = records_size;
    sge.lkey = records_mr->lkey;


    wr.wr_id = 2;
    wr.wr.rdma.remote_addr = remote_addr + sizeof(BucketHeader);


    if (ibv_post_send(ctx->connections[target_rank].qp, &wr, &bad_wr)) {
        debug_log("ERROR", "Failed to post records read", ctx->rank);
        ibv_dereg_mr(records_mr);
        free(records_buf);
        ibv_dereg_mr(header_mr);
        free(header_buf);
        return result;
    }


    do {
        ne = ibv_poll_cq(ctx->cq, 1, &wc);
    } while (ne == 0);

    if (ne < 0 || wc.status != IBV_WC_SUCCESS) {
        snprintf(debug_msg, sizeof(debug_msg),
                "Records read failed with status: %s", ibv_wc_status_str(wc.status));
        debug_log("ERROR", debug_msg, ctx->rank);
        ibv_dereg_mr(records_mr);
        free(records_buf);
        ibv_dereg_mr(header_mr);
        free(header_buf);
        return result;
    }

    WordRecord* records = static_cast<WordRecord*>(records_buf);
    for (uint64_t i = 0; i < header->record_count; i++) {
        if (strcmp(records[i].word, word.c_str()) == 0) {
            result = records[i];
            break;
        }
    }


    ibv_dereg_mr(records_mr);
    free(records_buf);
    ibv_dereg_mr(header_mr);
    free(header_buf);

    return result;
}


void modify_qp_to_init(struct ibv_qp *qp) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = 1;
    attr.pkey_index = 0;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | 
                          IBV_ACCESS_REMOTE_READ | 
                          IBV_ACCESS_REMOTE_WRITE;

    int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    int ret = ibv_modify_qp(qp, &attr, flags);
    if (ret) {
        std::cerr << "Failed to modify QP to INIT, error: " << ret << ", errno: " << errno << std::endl;
        exit(1);
    }
}

void modify_qp_to_rtr(struct ibv_qp *qp, uint32_t remote_qpn, uint16_t remote_lid, 
                     uint32_t remote_psn, int rank) {
    char debug_msg[256];
    snprintf(debug_msg, sizeof(debug_msg), 
             "Transitioning to RTR - remote_qpn=%u, remote_lid=%u, remote_psn=%u",
             remote_qpn, remote_lid, remote_psn);
    debug_log("QP", debug_msg, rank);

    struct ibv_qp_attr query_attr;
    struct ibv_qp_init_attr init_attr;
    if (ibv_query_qp(qp, &query_attr, IBV_QP_STATE, &init_attr) == 0) {
        snprintf(debug_msg, sizeof(debug_msg), 
                "Current QP state before RTR: %d", query_attr.qp_state);
        debug_log("QP", debug_msg, rank);
    }

    struct ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_1024;
    attr.dest_qp_num = remote_qpn;
    attr.rq_psn = remote_psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 0;
    attr.ah_attr.dlid = remote_lid;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = 1;

    int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MIN_RNR_TIMER | IBV_QP_MAX_DEST_RD_ATOMIC;

    int ret = ibv_modify_qp(qp, &attr, flags);
    if (ret) {
        snprintf(debug_msg, sizeof(debug_msg), 
                "Failed to modify QP to RTR - errno=%d (%s)", errno, strerror(errno));
        debug_log("ERROR", debug_msg, rank);
        exit(1);
    }


    if (ibv_query_qp(qp, &query_attr, IBV_QP_STATE, &init_attr) == 0) {
        snprintf(debug_msg, sizeof(debug_msg), 
                "QP state after RTR transition: %d", query_attr.qp_state);
        debug_log("QP", debug_msg, rank);
    }
}

void modify_qp_to_rts(struct ibv_qp *qp, uint32_t local_psn, int rank) {
    char debug_msg[256];
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));


    struct ibv_qp_attr query_attr;
    struct ibv_qp_init_attr init_attr;
    if (ibv_query_qp(qp, &query_attr, IBV_QP_STATE, &init_attr)) {
        debug_log("ERROR", "Failed to query QP state before RTS transition", rank);
        exit(1);
    }


    if (query_attr.qp_state != IBV_QPS_RTR) {
        snprintf(debug_msg, sizeof(debug_msg), 
                "QP not in RTR state (current state: %d)", query_attr.qp_state);
        debug_log("ERROR", debug_msg, rank);
        exit(1);
    }


    attr.qp_state = IBV_QPS_RTS;
    attr.path_mtu = IBV_MTU_1024;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = local_psn;
    attr.max_rd_atomic = 1;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;

    int flags = IBV_QP_STATE | 
                IBV_QP_TIMEOUT | 
                IBV_QP_RETRY_CNT |
                IBV_QP_RNR_RETRY | 
                IBV_QP_SQ_PSN | 
                IBV_QP_MAX_QP_RD_ATOMIC;

    int ret = ibv_modify_qp(qp, &attr, flags);
    if (ret) {
        snprintf(debug_msg, sizeof(debug_msg),
                "Failed to modify QP to RTS, error: %d, errno: %s", 
                ret, strerror(errno));
        debug_log("ERROR", debug_msg, rank);
        exit(1);
    }


    if (ibv_query_qp(qp, &query_attr, IBV_QP_STATE, &init_attr)) {
        debug_log("ERROR", "Failed to verify RTS transition", rank);
        exit(1);
    }

    if (query_attr.qp_state != IBV_QPS_RTS) {
        snprintf(debug_msg, sizeof(debug_msg),
                "Failed to transition to RTS (state: %d)", query_attr.qp_state);
        debug_log("ERROR", debug_msg, rank);
        exit(1);
    }

    debug_log("QP", "Successfully transitioned to RTS state", rank);
}


void setup_rdma_connections(RDMAContext* ctx, int rank, int size, MPI_Comm comm) {
    debug_log("CONN", "Starting RDMA connection setup", rank);


    struct ibv_port_attr port_attr;
    if (ibv_query_port(ctx->context, 1, &port_attr)) {
        debug_log("ERROR", "Failed to query port attributes", rank);
        exit(1);
    }

    char debug_msg[512];
    snprintf(debug_msg, sizeof(debug_msg), 
             "Port attributes: lid=%u, max_msg_sz=%u, active_width=%d, active_speed=%d",
             port_attr.lid, port_attr.max_msg_sz, port_attr.active_width, 
             port_attr.active_speed);
    debug_log("CONN", debug_msg, rank);

    srand48(time(NULL) * rank);
    uint32_t local_psn = lrand48() & 0xffffff;

    std::vector<QPInfo> all_qp_info(size * size);
    
    for (int i = 0; i < size; i++) {
        if (i != rank) {
            ctx->connections[i].qp = create_qp(ctx);
            if (!ctx->connections[i].qp) {
                debug_log("ERROR", "Failed to create QP", rank);
                exit(1);
            }
            
            QPInfo& qp_info = all_qp_info[rank * size + i];
            qp_info.qp_num = ctx->connections[i].qp->qp_num;
            qp_info.lid = port_attr.lid;
            qp_info.psn = local_psn;
            qp_info.addr = (uint64_t)ctx->mr->addr;
            qp_info.rkey = ctx->mr->rkey;

            
            modify_qp_to_init(ctx->connections[i].qp);
        }
    }

    MPI_Barrier(comm);

    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                  all_qp_info.data(), sizeof(QPInfo) * size, MPI_BYTE,
                  comm);

    MPI_Barrier(comm);

    for (int i = 0; i < size; i++) {
        if (i != rank) {
            QPInfo& remote_info = all_qp_info[i * size + rank];
            ctx->connections[i].remote_qpn = remote_info.qp_num;
            ctx->connections[i].remote_lid = remote_info.lid;
            ctx->connections[i].remote_psn = remote_info.psn;
            ctx->connections[i].remote_addr = remote_info.addr;
            ctx->connections[i].remote_rkey = remote_info.rkey;

            modify_qp_to_rtr(ctx->connections[i].qp,
                          ctx->connections[i].remote_qpn,
                          ctx->connections[i].remote_lid,
                          ctx->connections[i].remote_psn,
                          rank);

            MPI_Barrier(comm);  

            
            modify_qp_to_rts(ctx->connections[i].qp, local_psn, rank);

            snprintf(debug_msg, sizeof(debug_msg),
                    "Established connection with rank %d (qpn=%u lid=%u psn=%u)",
                    i, ctx->connections[i].remote_qpn,
                    ctx->connections[i].remote_lid,
                    ctx->connections[i].remote_psn);
            debug_log("CONN", debug_msg, rank);
        }
    }

    MPI_Barrier(comm);
    debug_log("CONN", "All RDMA connections established", rank);
}


void cleanup_rdma(RDMAContext* ctx) {
    if (ctx->connections) {
        for (int i = 0; i < 8; i++) {
            if (i != ctx->rank && ctx->connections[i].qp) {
                ibv_destroy_qp(ctx->connections[i].qp);
                ctx->connections[i].qp = nullptr;
            }
        }
        delete[] ctx->connections;
    }

    if (ctx->cq) {
        ibv_destroy_cq(ctx->cq);
        ctx->cq = nullptr;
    }

    if (ctx->comp_channel) {
        ibv_destroy_comp_channel(ctx->comp_channel);
        ctx->comp_channel = nullptr;
    }

    if (ctx->mr) {
        ibv_dereg_mr(ctx->mr);
        ctx->mr = nullptr;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (ctx->pd) {
        ibv_dealloc_pd(ctx->pd);
        ctx->pd = nullptr;
    }

    if (ctx->context) {
        ibv_close_device(ctx->context);
        ctx->context = nullptr;
    }
}


void check_rdma_error(const char* msg, bool condition) {
    if (condition) {
        std::cerr << "Error: " << msg << std::endl;
        std::cerr << "Error detail: " << strerror(errno) << std::endl;
        exit(1);
    }
}


bool post_rdma_read(RDMAContext* ctx, int target_rank, void* local_addr, size_t len, 
                   uint64_t remote_addr, uint32_t remote_rkey) {
    struct ibv_sge sg;
    struct ibv_send_wr wr;
    struct ibv_send_wr *bad_wr;

    memset(&sg, 0, sizeof(sg));
    sg.addr = reinterpret_cast<uint64_t>(local_addr);
    sg.length = len;
    sg.lkey = ctx->mr->lkey;

    memset(&wr, 0, sizeof(wr));
    wr.wr_id = 0;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.sg_list = &sg;
    wr.num_sge = 1;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = remote_rkey;

    return ibv_post_send(ctx->connections[target_rank].qp, &wr, &bad_wr) == 0;
}



bool wait_for_completion(RDMAContext* ctx) {
    struct ibv_wc wc;
    int num_completed;
    
    do {
        num_completed = ibv_poll_cq(ctx->cq, 1, &wc);
    } while (num_completed == 0);

    if (num_completed < 0) {
        return false;
    }

    return wc.status == IBV_WC_SUCCESS;
}


void query_words_rdma(const std::vector<std::string>& query_words, 
                     Bucket* local_buckets, 
                     int rank, 
                     int size, 
                     RDMAContext* ctx) {
    if (rank == 0) {
        std::cout << "\n====== ====== ====== ====== \n";
        std::cout << "   Starting the query ... \n";
        std::cout << "====== ====== ====== ====== \n";
    }


    std::vector<RDMAContext*> remote_contexts(size);
    

    if (rank == 0) {
        std::unordered_map<std::string, bool> processed_words;
        
        for (const std::string& query_word : query_words) {
            if (processed_words[query_word]) continue;
            processed_words[query_word] = true;

            unsigned int hash = hash_function(query_word);
            int target_rank = (hash >> 8) & 0x7;
            int bucket_number = hash & 0xFF;

            char debug_msg[256];
            snprintf(debug_msg, sizeof(debug_msg), 
                    "Querying word '%s' from rank %d, bucket %d",
                    query_word.c_str(), target_rank, bucket_number);
            debug_log("QUERY", debug_msg, rank);

            WordRecord found_record;
            if (target_rank == 0) {

                Bucket* bucket = &local_buckets[bucket_number];
                found_record = WordRecord(); 
                
                for (uint64_t i = 0; i < bucket->header.record_count; ++i) {
                    if (strcmp(bucket->records[i].word, query_word.c_str()) == 0) {
                        found_record = bucket->records[i];
                        break;
                    }
                }
            } else {
                
                found_record = rdma_read_record(ctx, target_rank, bucket_number, query_word);
            }

            std::cout << query_word << " - Freq: " << found_record.frequency;
            if (found_record.frequency > 0) {
                std::cout << "; Loc (<= 7): ";
                uint64_t loc_count = std::min(found_record.frequency, uint64_t(7));
                for (uint64_t i = 0; i < loc_count; ++i) {
                    std::cout << found_record.locations[i];
                    if (i != loc_count - 1) std::cout << " ";
                }
            }
            std::cout << std::endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}


std::vector<std::string> read_query_file(const std::string& filename) {
    std::vector<std::string> query_words;
    std::ifstream file(filename);
    std::string line;
    std::regex word_regex(R"([a-zA-Z0-9]{1,62})");
    
    while (std::getline(file, line)) {
        auto words_begin = std::sregex_iterator(line.begin(), line.end(), word_regex);
        auto words_end = std::sregex_iterator();
        
        for (auto it = words_begin; it != words_end; ++it) {
            query_words.push_back(to_uppercase(it->str()));
        }
    }
    return query_words;
}

void process_file(const std::string& filename, Bucket* local_buckets, int rank, int size, MPI_Comm comm) {
    std::ifstream infile(filename, std::ios::binary | std::ios::ate);
    if (!infile.is_open()) {
        if (rank == 0) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
        }
        MPI_Abort(comm, 1);
    }

    std::streampos file_size = infile.tellg();
    infile.seekg(0, std::ios::beg);

    uint64_t total_size = static_cast<uint64_t>(file_size);
    uint64_t chunk_size = total_size / size;
    uint64_t start_offset = rank * chunk_size;
    uint64_t end_offset = (rank == size - 1) ? total_size : start_offset + chunk_size;

    if (start_offset != 0) {
        infile.seekg(start_offset - 1);
        char ch;
        infile.get(ch);
        while (start_offset < end_offset && std::isalnum(ch)) {
            infile.get(ch);
            start_offset++;
        }
    }

    std::string buffer;
    buffer.reserve(chunk_size);
    infile.seekg(start_offset);

    char ch;
    while (infile.tellg() < static_cast<std::streampos>(end_offset) && infile.get(ch)) {
        buffer.push_back(ch);
    }
    infile.close();

    auto words = extract_words_with_indices(buffer, start_offset);

    std::vector<std::vector<WordOccurrence>> send_buffers(size);
    for (const auto& [word, index] : words) {
        std::string upper_word = to_uppercase(word);
        unsigned int hash = hash_function(upper_word);
        int process_id = (hash >> 8) & 0x7;

        WordOccurrence occurrence;
        memset(&occurrence, 0, sizeof(WordOccurrence));
        strncpy(occurrence.word, upper_word.c_str(), 63);
        occurrence.location = index;

        send_buffers[process_id].push_back(occurrence);
    }

    std::vector<int> send_counts(size);
    std::vector<int> send_displs(size, 0);
    for (int i = 0; i < size; ++i) {
        send_counts[i] = send_buffers[i].size() * sizeof(WordOccurrence);
    }

    std::vector<int> recv_counts(size);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);

    for (int i = 1; i < size; ++i) {
        send_displs[i] = send_displs[i - 1] + send_counts[i - 1];
    }

    std::vector<int> recv_displs(size, 0);
    for (int i = 1; i < size; ++i) {
        recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
    }

    std::vector<char> send_data;
    int total_send_size = send_displs[size - 1] + send_counts[size - 1];
    send_data.reserve(total_send_size);

    for (const auto& buffer : send_buffers) {
        const char* data = reinterpret_cast<const char*>(buffer.data());
        send_data.insert(send_data.end(), data, data + buffer.size() * sizeof(WordOccurrence));
    }

    int total_recv_size = recv_displs[size - 1] + recv_counts[size - 1];
    std::vector<char> recv_data(total_recv_size);

    MPI_Alltoallv(send_data.data(), send_counts.data(), send_displs.data(), MPI_CHAR,
                  recv_data.data(), recv_counts.data(), recv_displs.data(), MPI_CHAR, comm);

    int num_occurrences = total_recv_size / sizeof(WordOccurrence);
    WordOccurrence* occurrences = reinterpret_cast<WordOccurrence*>(recv_data.data());

    for (int i = 0; i < num_occurrences; ++i) {
        WordOccurrence& occurrence = occurrences[i];
        std::string upper_word = occurrence.word;
        unsigned int hash = hash_function(upper_word);
        int bucket_number = hash & 0xFF;

        Bucket* bucket = &local_buckets[bucket_number];
        uint64_t record_count = bucket->header.record_count;

        WordRecord* found_record = nullptr;
        for (uint64_t j = 0; j < record_count; ++j) {
            if (strcmp(bucket->records[j].word, upper_word.c_str()) == 0) {
                found_record = &bucket->records[j];
                break;
            }
        }

        if (found_record) {
            found_record->frequency += 1;
            if (found_record->frequency <= 7) {
                found_record->locations[found_record->frequency - 1] = occurrence.location;
            }
        } else if (record_count < 32767) {
            WordRecord& new_record = bucket->records[record_count];
            strncpy(new_record.word, upper_word.c_str(), 63);
            new_record.frequency = 1;
            new_record.locations[0] = occurrence.location;
            bucket->header.record_count += 1;
        }
    }
}

void query_words(const std::vector<std::string>& query_words, Bucket* local_buckets, int rank, int size, MPI_Comm comm) {
    if (rank == 0) {
        std::cout << "\n====== ====== ====== ====== \n";
        std::cout << "   Starting the query ... \n";
        std::cout << "====== ====== ====== ====== \n";
    }
    MPI_Barrier(comm);

    std::unordered_map<std::string, bool> processed_words;
    
    for (const std::string& query_word : query_words) {
        if (processed_words[query_word]) continue;
        processed_words[query_word] = true;

        WordRecord found_record;
        memset(&found_record, 0, sizeof(WordRecord));
        
        unsigned int hash = hash_function(query_word);
        int target_rank = (hash >> 8) & 0x7;
        int bucket_number = hash & 0xFF;

        if (rank == target_rank) {
            Bucket* bucket = &local_buckets[bucket_number];
            for (uint64_t i = 0; i < bucket->header.record_count; ++i) {
                if (strcmp(bucket->records[i].word, query_word.c_str()) == 0) {
                    found_record = bucket->records[i];
                    break;
                }
            }
        }

        if (rank == target_rank && rank != 0) {
            MPI_Send(&found_record, sizeof(WordRecord), MPI_CHAR, 0, 0, comm);
        }
        else if (rank == 0 && target_rank != 0) {
            MPI_Recv(&found_record, sizeof(WordRecord), MPI_CHAR, target_rank, 0, comm, MPI_STATUS_IGNORE);
        }

        if (rank == 0) {
            std::cout << query_word << " - Freq: " << found_record.frequency;
            if (found_record.frequency > 0) {
                std::cout << "; Loc (<= 7): ";
                uint64_t loc_count = std::min(found_record.frequency, uint64_t(7));
                for (uint64_t i = 0; i < loc_count; ++i) {
                    std::cout << found_record.locations[i];
                    if (i != loc_count - 1) std::cout << " ";
                }
            }
            std::cout << std::endl;
        }

        MPI_Barrier(comm);
    }
}


void display_results(Bucket* local_buckets, int rank, int size, MPI_Comm comm) {
    WordRecord max_record;
    uint64_t max_frequency = 0;

    for (int i = 0; i < 256; ++i) {
        Bucket* bucket = &local_buckets[i];
        for (uint64_t j = 0; j < bucket->header.record_count; ++j) {
            WordRecord* record = &bucket->records[j];
            if (record->frequency > max_frequency) {
                max_frequency = record->frequency;
                max_record = *record;
            }
        }
    }

    if (rank != 0) {
        MPI_Send(&max_frequency, 1, MPI_UINT64_T, 0, 0, comm);
        if (max_frequency > 0) {
            MPI_Send(&max_record, sizeof(WordRecord), MPI_CHAR, 0, 1, comm);
        }
    } else {
        if (max_frequency > 0) {
            std::cout << "Rank 0: " << max_record.word 
                     << " - Freq: " << max_record.frequency 
                     << "; Loc (<= 7): ";
            uint64_t loc_count = std::min(max_record.frequency, uint64_t(7));
            for (uint64_t j = 0; j < loc_count; ++j) {
                std::cout << max_record.locations[j];
                if (j != loc_count - 1) std::cout << " ";
            }
            std::cout << std::endl;
        }

        for (int i = 1; i < size; i++) {
            uint64_t remote_freq;
            MPI_Recv(&remote_freq, 1, MPI_UINT64_T, i, 0, comm, MPI_STATUS_IGNORE);
            
            if (remote_freq > 0) {
                WordRecord remote_record;
                MPI_Recv(&remote_record, sizeof(WordRecord), MPI_CHAR, i, 1, comm, MPI_STATUS_IGNORE);
                
                std::cout << "Rank " << i << ": " << remote_record.word 
                         << " - Freq: " << remote_record.frequency 
                         << "; Loc (<= 7): ";
                uint64_t loc_count = std::min(remote_record.frequency, uint64_t(7));
                for (uint64_t j = 0; j < loc_count; ++j) {
                    std::cout << remote_record.locations[j];
                    if (j != loc_count - 1) std::cout << " ";
                }
                std::cout << std::endl;
            }
        }
    }
    MPI_Barrier(comm);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 8) {
        if (rank == 0) {
            std::cerr << "This program requires exactly 8 MPI processes." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    if (argc != 3) {
        if (rank == 0) {
            std::cerr << "Usage: mpirun -np 8 " << argv[0] << " <input_file> <query_file>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    std::string input_file = argv[1];
    std::string query_file = argv[2];

    size_t total_size = sizeof(Bucket) * 256;
    Bucket* local_buckets = static_cast<Bucket*>(aligned_alloc_wrapper(4096, total_size));
    if (!local_buckets) {
        std::cerr << "Failed to allocate aligned memory for buckets" << std::endl;
        MPI_Finalize();
        return 1;
    }
    std::memset(local_buckets, 0, total_size);
    RDMAContext rdma_ctx = {};
    init_rdma(&rdma_ctx, local_buckets, rank);

    setup_rdma_connections(&rdma_ctx, rank, size, MPI_COMM_WORLD);

    process_file(input_file, local_buckets, rank, size, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    display_results(local_buckets, rank, size, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<std::string> query_word_list;
    if (rank == 0) {
        query_word_list = read_query_file(query_file);
    }

    query_words_rdma(query_word_list, local_buckets, rank, size, &rdma_ctx);

    cleanup_rdma(&rdma_ctx);
    free(local_buckets); 

    MPI_Finalize();
    return 0;
}