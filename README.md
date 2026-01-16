# Device Orchestrator

A modular Python framework for loading YAML-based automation recipes, executing them on pluggable device objects, and generating structured logs.

## Features

- **YAML-based recipes**: Define devices and automation steps declaratively
- **Pluggable device architecture**: Factory pattern with abstract base interface
- **Simulated devices**: Built-in simulation with noise, latency, drift, and fault injection
- **Async execution**: Full async/await support for concurrent operations
- **Timeout enforcement**: Step-level timeout controls
- **Value validation**: Expected range checking for measurements
- **Structured logging**: JSONL events, CSV summaries, and JSON manifests
- **Deterministic testing**: Seeded RNG for reproducible simulations

## Project Structure

```
device-orchestrator/
├─ README.md
├─ requirements.txt
├─ src/
│  ├─ main.py              # CLI entry point
│  ├─ cli.py               # Command-line interface
│  ├─ config/
│  │  └─ examples/
│  │     └─ simple_demo.yml
│  ├─ devices/
│  │  ├─ __init__.py
│  │  ├─ base.py           # Abstract device interface
│  │  ├─ simulated.py      # Simulated device implementation
│  │  └─ factory.py        # Device factory
│  ├─ orchestrator/
│  │  ├─ __init__.py
│  │  ├─ runner.py         # Recipe execution engine
│  │  └─ logging_utils.py  # Structured logging
│  └─ utils/
│     └─ helpers.py        # Utility functions
└─ tests/
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run a recipe:

```bash
python -m src.main run src/config/examples/simple_demo.yml --out out/run1
```

With custom seed for reproducibility:

```bash
python -m src.main run src/config/examples/simple_demo.yml --out out/run1 --seed 42
```

## Recipe Format

```yaml
name: simple_demo

devices:
  heater:
    type: simulated
    params:
      latency_mean: 0.05
      noise_std: 0.02

steps:
  - name: set_heater
    device: heater
    cmd: set
    params: { value: 0.7 }

  - name: read_heater
    device: heater
    cmd: read
    expect: [0.6, 0.8]
    timeout: 1.0
```

## Output Files

- **events.jsonl**: One JSON object per line with step execution details
- **summary.csv**: CSV with timestamp, device, value, status columns
- **manifest.json**: Metadata about the recipe run

## License

MIT