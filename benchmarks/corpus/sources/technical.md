# Technical Prose Corpus

Technical writing about distributed systems, GPU execution, caching, software
architecture, scheduling, data structures, and networking — the kind of prose a
model processes in engineering workloads.

---

## Consistency in distributed systems

A distributed system that replicates data must choose, explicitly or by
accident, how much divergence it is willing to tolerate. Strong consistency
simplifies the mental model for every caller: reads observe the latest completed
write, and the system behaves like a single machine. The cost is coordination.
Every write that must be globally ordered pays a round trip to a quorum, and
that round trip is bounded below by the speed of light between the fastest and
the slowest replica you are willing to wait for.

Most systems therefore move the choice to the data itself. Keys with strict
invariants — account balances, inventory counts, leader elections — go through
consensus protocols such as Raft or multi-Paxos, where a leader serializes
appends and followers replicate a write-ahead log. Keys without strict
invariants — feeds, counters, caches, session state — accept eventual
consistency and resolve conflicts with last-writer-wins timestamps, version
vectors, or application-specific merge functions.

The subtle failure mode is not the outage; it is the quiet window where two
replicas disagree and both are serving reads. Applications that assume
read-your-writes consistency after a failover discover, at the worst possible
moment, that the assumption was load-bearing. This is why serious systems make
consistency levels part of the request API rather than a property of the
cluster: the caller knows which operations need ordering guarantees, and the
storage layer knows which operations can be relaxed.

## GPU execution fundamentals

A modern GPU is a throughput machine organized around thousands of small
threads grouped into warps or wavefronts. The unit of scheduling is the warp,
and the fundamental constraint is that all lanes of a warp execute the same
instruction at the same time. Divergent branches do not crash; they serialize,
executing each path while masking the other, which is why branchy kernels
degrade gracefully rather than catastrophically.

Memory behavior dominates kernel performance. Arithmetic intensity — the ratio
of floating point operations to bytes moved — decides whether a kernel is
compute-bound or bandwidth-bound. Matrix multiplication with large tiles is
comfortably compute-bound because each loaded value participates in many
multiply-adds. Element-wise operations are bandwidth-bound, and no amount of
instruction tuning will make them faster than the DRAM bus allows.

Latency hiding is the other half of the story. When a warp issues a load, it
does not stall the whole processor; the scheduler swaps in another ready warp
while the data is in flight. Occupancy — how many warps can be resident per
multiprocessor — is therefore a means, not an end. Enough warps to cover memory
latency is sufficient; beyond that, more occupancy only competes for registers
and shared memory. The fastest kernels tend to be the ones that measure first,
keep tiles in fast memory, and make the common path boring.

## Caching strategies

A cache is a bet that the future will resemble the past. The bet pays off when
access patterns are skewed — a small working set absorbs most requests — and it
costs when the working set shifts faster than the cache can adapt. Everything
else is bookkeeping.

Write policies decide where staleness lives. Write-through caches keep the
replica accurate at the cost of doubling write latency. Write-back caches
acknowledge quickly and flush lazily, trading durability risk for speed.
Read-through caches hide the miss path behind the cache interface, while
cache-aside caches leave the lookup to the caller and keep the cache dumb.

Eviction policy is where caches are won and lost. LRU is the default and is
fine for scan-resistant workloads, but a single sequential scan can evict the
entire hot set. Variants like LRU-K, ARC, and TinyLFU track frequency as well
as recency, at the cost of metadata and more complex invariants. In inference
systems, the cache unit is often not a key but a prefix: reusable computation
for identical token sequences, keyed by a hash of the sequence, with eviction
measured in bytes of intermediate state rather than number of entries.

## Scheduling

A scheduler answers one question repeatedly: which work runs now? The answer
depends on what the scheduler is optimizing. Batch schedulers maximize
throughput and accept latency variance. Interactive schedulers minimize tail
latency and accept lower utilization. Mixed schedulers try to carve the
difference with priority classes, quotas, and preemption.

The cost of a scheduling decision is never zero. Every policy that considers
future state — shortest remaining time, cost-aware placement, deadline
feasibility — pays for the estimation. Simple policies make fast, predictable
decisions; complex policies make better decisions on average and pathological
ones when the estimator is wrong. Production systems usually win by making the
fast path simple and escalating to the complex path only for the requests that
need it.

Starvation is the quiet failure of any priority scheme. Without anti-starvation
mechanisms — aging, reservations, bounded bypass — low-priority work can wait
indefinitely under sustained high-priority load. The result is a system that
looks healthy on average latency while a minority of requests experiences
unbounded delay, which is exactly what tail percentiles are for.

## Data structures for hot paths

Choosing a data structure for a hot path is less about asymptotic notation and
more about memory layout. A hash map with open addressing and compact keys will
beat a theoretically superior tree when the working set fits in cache lines,
because the constant factor is the CPU waiting on DRAM. Pointer chasing is the
enemy; contiguous arrays with integer indices are the friend.

Concurrency changes the calculus again. Lock-free structures promise progress
under contention but deliver complexity: compare-and-swap loops, memory
ordering arguments, and ABA hazards that surface once a week in production.
Often the winning design is sharding — a striped array of ordinary locked maps —
because contention drops below the point where cleverness matters.

The boring recommendation stands: measure with production-shaped data, keep the
common case contiguous, and reserve specialized structures for the places the
profiler actually points to.

## Networking and congestion

Reliable transport over an unreliable network is a story of feedback. TCP
congestion control probes for available bandwidth with additive increase and
backs off with multiplicative decrease when loss appears, which converts a
shared bottleneck into a roughly fair allocation over time. Datacenter variants
tune the parameters aggressively — larger initial windows, finer round-trip
measurement, explicit congestion notification from switches — but the loop
stays the same: send, observe, adjust.

Latency budgets compound. A request that touches four services pays four
network round trips plus queueing at each hop, and tail latency is dominated by
the slowest dependency rather than the average one. Hedged requests — issuing a
duplicate to a second replica after a delay near the tail — trade a little
capacity for a lot of p99, and work best where requests are cheap and
independent.

Bufferbloat remains the classic misconfiguration: oversized queues convert
congestion into latency instead of loss, making the network feel slow while
looking idle. Modern queue management — fair queuing, active queue management,
byte-limited buffers — keeps standing queues short so flows see congestion
early and respond before users notice.

## Observability

Metrics tell you that something changed; traces tell you where; logs tell you
why. A system with only one of the three is a system that will be debugged by
rumor. The practical goal is correlation: a spike in latency visible in
metrics, a trace showing which span grew, and a log line explaining what the
component was doing at that moment.

Cardinality is the tax on detail. Every label on a metric multiplies the number
of time series, and unbounded labels — user ids, request ids, file paths — can
take down the monitoring system faster than the incident being monitored. Good
practice puts high-cardinality identifiers in traces, keeps metric labels to a
bounded vocabulary, and treats logs as an indexed archive rather than a data
stream.

Sampling is the other lever. Full-fidelity tracing is ideal and unaffordable at
scale; head-based sampling with a small rate, tail-based sampling that keeps
interesting traces, or adaptive sampling that raises the rate during incidents
all beat the naive choice. The question is never whether to sample, but which
information must survive the sample.
