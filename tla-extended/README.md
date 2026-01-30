# Code-Level TLA+ Specification for etcd/raft Core Module

This directory contains a code-level TLA+ specification for the etcd/raft library, along with tooling for trace validation and model checking.

## Overview

The specification models the core Raft algorithm as implemented in etcd/raft, including:

- **Leader election** - RequestVote handling, term management
- **Log replication** - AppendEntries, log matching, commitment
- **Membership reconfiguration** - Joint consensus, ConfChangeV2, single-node changes
- **Snapshot** - Snapshot sending/receiving, log compaction
- **Progress tracking** - StateProbe, StateReplicate, StateSnapshot, Inflights
- **Flow control** - MsgAppFlowPaused, MaxInflightMsgs

## Directory Structure

```
tla-extended/
├── etcdraft.tla          # Main TLA+ specification
├── etcdraft.cfg          # Base TLC configuration
├── MCetcdraft.tla        # Model checking configuration
├── MCetcdraft.cfg        # TLC model checking settings
├── Traceetcdraft.tla     # Trace validation specification
├── Traceetcdraft.cfg     # Trace validation settings
├── harness/              # Go test harness for trace generation
│   ├── main.go           # Harness entry point
│   ├── parser.go         # Trace parser
│   └── rawnode_access.go # RawNode accessor
├── patches/              # Instrumentation patches
│   └── instrumentation.patch
└── traces/               # 37 trace files covering various scenarios
    ├── basic.ndjson
    ├── confchange_*.ndjson    # Configuration change scenarios
    ├── snapshot_*.ndjson      # Snapshot scenarios
    └── ...
```

## Trace Validation

### 1. Apply Instrumentation Patch

```bash
cd /path/to/raft
git apply tla-extended/patches/instrumentation.patch
```

### 2. Generate Traces

Build and run the harness to generate traces from raft tests:

```bash
cd tla-extended/harness
go build -o harness .
./harness -test TestBasic -output ../traces/
```

### 3. Validate Traces

Use `validate.sh` to validate traces in batch:

```bash
cd tla-extended
./validate.sh \
    -s Traceetcdraft.tla \
    -c Traceetcdraft.cfg \
    traces/*.ndjson
```

By default, `validate.sh` uses all CPU cores. Use `-p <num>` to limit concurrency.

## Trace Validation Coverage

The specification has been validated against 37 traces generated from etcd/raft's test suite:

| Metric | Value |
|--------|-------|
| Total statements | 431 |
| Covered statements | 407 |
| Coverage | **94.43%** |

The uncovered statements (5.57%) represent edge cases that require strict timing sequences to trigger. The current test harness cannot generate such scenarios, as they involve specific interleavings that are difficult to produce through standard testing. Future improvements to the harness could potentially increase coverage.

## Invariants

The specification defines **85 invariants** organized into 8 categories, derived from 3 sources:

### Sources

| Source | Description |
|--------|-------------|
| **Raft Paper** | Safety properties from the Raft consensus paper |
| **Code** | Invariants derived from etcd/raft implementation details |
| **Issue/Bug** | Regression tests for historical bugs from git history |

**Categories:**

| Category | Count | Example Invariants |
|----------|-------|-------------------|
| Log | 17 | `LogMatchingInv`, `QuorumLogInv`, `CommitIndexBoundInv` |
| Snapshot | 14 | `SnapshotStateInv`, `SnapshotTermValidInv`, `SnapshotPendingInv` |
| Progress | 13 | `ProbeLimitInv`, `MatchIndexBoundInv`, `NextIndexBoundInv` |
| Inflights | 11 | `InflightsLogIndexInv`, `InflightsMonotonicInv`, `InflightsInv` |
| Config | 9 | `JointConfigNonEmptyInv`, `ConfigurationInv`, `SinglePendingLeaveJointInv` |
| Message | 8 | `MessageIndexValidInv`, `AppendEntriesCommitBoundInv`, `MessageEndpointsValidInv` |
| Election | 7 | `MoreThanOneLeaderInv`, `VotesGrantedSubsetInv`, `CandidateVotedForSelfInv` |
| Term | 6 | `TermPositiveInv`, `LeaderTermPositiveInv`, `TermAndVoteInv` |

## Model Checking (Simulation)

For quick exploration via simulation:

```bash
java -XX:+UseParallelGC \
    -cp tla2tools.jar:CommunityModules-deps.jar \
    tlc2.TLC \
    -config MCetcdraft.cfg \
    MCetcdraft.tla \
    -simulate \
    -depth 100
```
