# ResQ-Stream: Member 1 (Data and Streaming)

This directory holds Member 1's deliverables for the ResQ-Stream pipeline:

1. The data contract (frozen Day 1, shared by all four members)
2. The supply chain simulator
3. Kafka topic design, producer and consumer
4. The Db2 schema for inventory and orders
5. Local infrastructure (Docker) and tests

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest tests -q                              # contract and simulator tests
mkdir -p data
python -m sim.simulator --hours 168 --seed 42 --jsonl data/run_42.jsonl   # raw event stream
python -m sim.export_csv --hours 336 --out data/csv                       # CSVs for Member 2

docker compose up -d kafka                             # local Kafka on localhost:9092
python -m streaming.create_topics
python -m streaming.producer --hours 168 --pace 0.05   # stream into Kafka
python -m streaming.consumer rq.inventory.telemetry rq.demand.signals --from-beginning --max 200
```

`python -m streaming.producer --dry-run` validates and counts messages without a broker.

### Connecting to IBM Event Streams

Set these variables. Nothing in the code changes:

```bash
export KAFKA_BOOTSTRAP_SERVERS=<broker list from the Event Streams console>
export KAFKA_SECURITY_PROTOCOL=SASL_SSL
export KAFKA_SASL_USERNAME=token
export KAFKA_SASL_PASSWORD=<Event Streams API key>
export KAFKA_REPLICATION=3                             # only if the cluster allows it
```

## Layout

| Path | Purpose |
|---|---|
| `contracts/resq_messages.schema.json` | Message contract v1.0 (JSON Schema) |
| `contracts/validate.py` | Validation helper used by producer, consumer and tests |
| `sim/simulator.py` | SimPy simulator: hospitals, warehouses, distributors, lanes, disruptions |
| `sim/export_csv.py` | Writes `telemetry_train.csv`, `telemetry_test.csv` and `edges.csv` for Member 2 |
| `streaming/topics.py` | Topic names, partitions, retention, routing and partition keys |
| `streaming/config.py` | Kafka client config (local PLAINTEXT or Event Streams SASL_SSL) |
| `streaming/producer.py` | Runs the simulator and publishes validated messages |
| `streaming/consumer.py` | Consumes topics; template for Members 2 and 3 |
| `streaming/create_topics.py` | Creates topics from `topics.py` |
| `db2/schema.sql` | Db2 for z/OS DDL for inventory, shipments, disruptions, recommendations |
| `docker-compose.yml`, `Dockerfile` | Local Kafka and containerised producer |
| `tests/test_contract.py` | Contract compliance, determinism, rejection tests |
| `tests/test_streaming_mocked.py` | Producer, consumer and topic code against fake clients |
| `tests/test_db2_ddl.py` | Db2 DDL constraints, checked via SQLite approximation |
| `tests/test_export_csv.py` | CSV columns, label definition, edges, shared network |
| `docs/judge_qa_ibm_z.md` | Shared talking points on the IBM Z story |

## Data contract (v1.0)

| Topic | Key | Message type | Emitted |
|---|---|---|---|
| `rq.inventory.telemetry` | `node_id` | `inventory_telemetry` | Every node and item, every simulated hour |
| `rq.demand.signals` | `node_id` | `demand_signal` | Every hospital and item, every hour |
| `rq.shipments.status` | `dst_node` | `shipment_status` | Dispatched, delayed, delivered |
| `rq.disruptions` | `target_id` | `disruption` | Once, when a disruption starts |
| `rq.deadletter` | none | n/a | Any message that fails validation |

Telemetry fields: `node_id`, `item_id`, `stock_level` (int ≥ 0), `consumption_rate`
(units/hour, 0.0 for warehouses), `in_transit_qty`. Every message carries `schema_version`,
`message_type`, `event_id` (UUID) and `event_time` (UTC, `...Z`).

**Changing the contract** needs all four members to agree, a version bump in
`schema_version`, and an update to this README. Do not change fields in place.

## CSV files for Member 2

`python -m sim.export_csv --hours 336 --out data/csv` writes:

| File | Contents |
|---|---|
| `telemetry_train.csv` | One row per node, item and hour from the training run (seed 42) |
| `telemetry_test.csv` | Same columns from a separate run (seed 7). Same network. |
| `edges.csv` | `src`, `dst`, `transit_hrs` for each lane |

Columns: `hour, node_id, node_type, item, stock, consumption_rate, inbound_eta_hrs,
event_flag, stockout_now, stockout_next_N`.

Conventions Member 2 needs to know:

- **`stockout_next_N`** is 1 if the item is out of stock at any of the next 6 hours (`N_HOURS`
  in `sim/export_csv.py`, which must match Member 2's config). It is the label. The last 6
  hours of each run are dropped, because their label is missing future data.
- **`inbound_eta_hrs`** is hours until the earliest inbound shipment lands. It is `-1.0`
  when nothing is in flight. Confirm this convention with Member 2 before training.
- **`event_flag`** is 1 while a disruption touches the node: a surge, a supplier delay, or a
  closed lane touching it.
- **`node_type`** includes `distributor`, as well as `hospital` and `warehouse`.

## Default scenario

| ID | Kind | Target | Window (sim hours) | Effect |
|---|---|---|---|---|
| DIS-001 to DIS-004 | outbreak_surge | each hospital | 48–200 | Demand ×6 |
| DIS-005 | supplier_delay | DIST-Central | 72–84 | +6 h on lanes leaving DIST-Central |
| DIS-006 | route_closure | WH-North__HOSP-02 | 120–144 | Lane closed; shipments wait |

The surge was tuned so stockouts are common enough to train on. The earlier scenario,
with one hospital surging, gave only about 0.1% positive labels. The current scenario
gives about 7% in training and 8% in test.

`run()` returns `stockout_hours`, `dispatched`, `delivered` and `unmet_orders`. Use
`stockout_hours` as the ground truth for the no-rerouting baseline in the evaluation.

## Handoffs

| When | From → To | What |
|---|---|---|
| Day 1 | Member 1 → 2, 3, 4 | Contract frozen (this schema). Each member confirms by reviewing the PR. |
| Mid-point | Member 1 → 2 | `data/csv/*.csv` from `sim.export_csv`, plus `data/*.jsonl` from the simulator |
| Mid-point | Member 1 → 3 | Live topic on Kafka. Member 3 builds the inference consumer. |
| Later | Member 3 → 4 | Risk scores. Member 4 writes recommendations to `TRANSFER_RECOMMENDATION`. |

Fallback: if Kafka is unavailable during the event, the `--jsonl` replay keeps the rest
of the pipeline moving. Member 3 can read the same files.

## What Member 1 must know by heart

- The data contract: field names, types, and what each topic means
- Partition keys and why they preserve per-node ordering
- How the simulator produces each disruption, and what stockout means in the data
- What Kafka topics, producers and consumers are, and the difference between a topic
  and a partition
- The IBM Z story in `docs/judge_qa_ibm_z.md`

## Learning path

- Python: SimPy process and timeout basics (simpy.readthedocs.io), then JSON Schema
- Kafka: the quickstart and concepts pages (kafka.apache.org/documentation)
- Db2 and SQL: CREATE TABLE, CHECK constraints, identity columns, indexes
- Docker: `docker compose up`, `docker compose logs`, the profile option
- Git: branch, commit, pull request. Keep schema changes in their own PR.

## Known limitations

- The simulator is not calibrated to real hospital data. It is for demonstrating the
  pipeline and the evaluation method, not for real supply planning.
- Warehouses are effectively unlimited in v1, so the optimiser never runs out of
  supply at source.
- The default scenario is tuned for label balance, not for realism. A model that scores
  well on it has learned this simulator, not hospital behaviour.
- The train and test CSVs share one network but use different noise. They do not test
  generalisation to a different supply chain.
- Db2 is not run locally. `tests/test_db2_ddl.py` loads the DDL into SQLite to check
  constraints, which is an approximation. Run `db2/schema.sql` on the datathon's Db2 for
  z/OS environment before the demo.
- Kafka is not run locally by the tests. `tests/test_streaming_mocked.py` checks the
  producer, consumer and topic code against fake clients. A real broker run
  (`docker compose up -d kafka`, then the producer) is still required before the demo.
- The Docker image has not been built in this environment, because the base image
  cannot be pulled from here.
- `confluent-kafka` wheels for s390x (IBM Z) may need building from source. That is
  Member 3's problem to confirm early.
