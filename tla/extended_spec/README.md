# System-Level TLA+ Specification for etcd/raft

This directory contains a system-level TLA+ specification for the etcd/raft library, extending the protocol-level specification in `../`. The specification models implementation details including progress tracking (`StateProbe`, `StateReplicate`, `StateSnapshot`), flow control (`MsgAppFlowPaused`, Inflights), detailed snapshot handling, and joint consensus configuration changes.

## Trace Validation

### 1. Apply Instrumentation Patch

```bash
cd /path/to/raft
git apply tla/extended_spec/patches/instrumentation.patch
```

### 2. Generate Traces

Build and run the harness to generate traces from raft tests:

```bash
cd tla/extended_spec/harness
go build -o harness .
./harness -test TestBasic -output ./traces/
```

### 3. Validate Traces

```bash
cd tla/extended_spec
./validate.sh -s Traceetcdraft.tla -c Traceetcdraft.cfg traces/*.ndjson
```

## Trace Validation Coverage

Validated against 37 traces from etcd/raft's test suite:
- **Coverage:** 94.43% (407/431 statements)

## Model Checking (Simulation)

```bash
java -XX:+UseParallelGC \
    -cp tla2tools.jar:CommunityModules-deps.jar \
    tlc2.TLC \
    -config MCetcdraft.cfg \
    MCetcdraft.tla \
    -simulate \
    -depth 100
```

## Invariants

The specification defines **88 invariants** derived from three sources:
- **Raft Paper:** Safety properties
- **Code:** Implementation details
- **Issue/Bug:** Historical bug regression tests 
