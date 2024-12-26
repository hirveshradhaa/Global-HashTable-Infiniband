# Global Distributed Hash Table for Word Counting

## Project Overview
This project implements a  Global Distributed Hash Table  (GDHT) to count word occurrences in a given data file. The program is designed to run in parallel on a cluster of machines, leveraging distributed computing and Remote Direct Memory Access (RDMA) operations. This solution is particularly useful for processing large-scale text files (up to 1GB) efficiently.

## Key Features

1.  Distributed Word Count :
   - Processes words from a data file in parallel using 8 processes.
   - Words are normalized to uppercase and counted regardless of case.
   - Implements a custom distributed hash table across 8 processes, each maintaining a local hash table.

2.  Query Capability :
   - A query file containing words is used to retrieve occurrence details (frequency and positions) from the GDHT.
   - RDMA read operations are used for efficient remote data access.

3.  Data Scalability :
   - Handles data sizes ranging from 1MB to 1GB.
   - Ensures scalability with performance analysis across different data sizes and configurations.

## Implementation Details

1.  Hash Table Design :
   - Each local hash table contains 256 buckets, with 4MB allocated per bucket.
   - Each word record consists of:
     - Word: 1–62 characters in length.
     - Frequency: 8-byte integer tracking occurrences.
     - Locations: Array of the first 7 occurrences.
   - Words are hashed using a custom function to determine their host process and bucket.

2.  Parallel Processing :
   - The program runs on 8 processes distributed across 2 or 4 nodes of a cluster.
   - RDMA write and atomic operations ensure concurrency control during updates.

3.  Query Execution :
   - Queries are performed by one designated process, which reads word records remotely using RDMA.
   - Outputs include the most frequent word from each process and details for each queried word.

## Requirements

### Software and Languages
-  Programming Language : C/C++
-  Cluster Computing : InfiniBand-enabled nodes
-  Build System : Makefile

### Input and Output
- Input:
  - Data files (e.g., `test1.txt`, `test2.txt`, up to 1GB).
  - Query file (`query.txt`).
- Output:
  - Word frequency and location details.
  - Most frequent word from each process.

### Performance Analysis
- Test cases:
  - Data file sizes: 1MB, 4MB, 64MB, 256MB, 512MB.
  - Performance results include average execution times over three runs.

## Running the Project

### Compilation
Use the provided `Makefile` to compile the program:
```bash
make
```

### Execution
Run the program with appropriate input files and cluster configurations. Example:
```bash
mpirun -np 8 ./word_count query.txt test1.txt
```

### Output
The program generates outputs for:
- Most frequent words from each local hash table.
- Query results for words in the input query file.

## Submission Structure
The project includes the following files:
- Source code files.
- `Makefile` for compilation.
- Performance analysis report (`analysis.pdf`).
- Test data files and output samples.

## Notes
- Ensure proper file permissions and directory setup as instructed.
- Refer to the project report for performance insights and implementation details.
