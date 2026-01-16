# Device Orchestrator - Quick Start Guide

## Installation

1. **Clone/navigate to the repository:**
   ```bash
   cd device-orchestrator
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running Your First Recipe

### Simple Demo

Run the included simple demo recipe:

```bash
python -m src.main run src/config/examples/simple_demo.yml
```

This will:
- Create 2 simulated devices (heater and sensor)
- Execute 6 automation steps
- Generate logs in the `out/` directory

### With Custom Output Directory

```bash
python -m src.main run src/config/examples/simple_demo.yml --out results/my_run
```

### With Deterministic Seed

For reproducible results (useful for testing):

```bash
python -m src.main run src/config/examples/simple_demo.yml --seed 42
```

## Understanding the Output

After running a recipe, you'll find three files in the output directory:

### 1. `events.jsonl` - Detailed Event Log

One JSON object per line containing full details of each step:

```json
{
  "timestamp": "2025-12-10T14:27:26.712342Z",
  "step_name": "set_heater",
  "device": "heater",
  "command": "set",
  "params": {"value": 0.7},
  "result": {"status": "ok", "value": 0.7},
  "latency": 0.054
}
```

### 2. `summary.csv` - Tabular Summary

Simplified view for data analysis:

```csv
timestamp,device,value,status
2025-12-10T14:27:26.712342Z,heater,0.7,ok
2025-12-10T14:27:26.769189Z,heater,0.714,ok
```

### 3. `manifest.json` - Execution Metadata

High-level summary of the run:

```json
{
  "recipe_name": "simple_demo",
  "seed": 42,
  "start_time": "2025-12-10T14:27:26.609349Z",
  "end_time": "2025-12-10T14:27:27.033293Z",
  "duration_seconds": 0.424,
  "success": true,
  "total_steps": 6,
  "failed_steps": 0
}
```

## Creating Your Own Recipe

### Recipe Structure

A recipe YAML file has two main sections:

```yaml
name: my_recipe

devices:
  my_device:
    type: simulated
    params:
      latency_mean: 0.05      # Average command time (seconds)
      noise_std: 0.02         # Measurement noise (std dev)
      drift_rate: 0.0         # Linear drift per second
      fail_rate: 0.0          # Failure probability (0-1)
      initial_value: 0.0      # Starting value

steps:
  - name: step_name
    device: my_device
    cmd: set                  # or "read" or "info"
    params: { value: 100 }    # optional parameters
    expect: [90, 110]         # optional range check
    timeout: 2.0              # max execution time
```

### Supported Commands

All simulated devices support:

- **`set`** - Write a value
  ```yaml
  cmd: set
  params: { value: 42.0 }
  ```

- **`read`** - Read current value (with noise)
  ```yaml
  cmd: read
  ```

- **`info`** - Get device metadata
  ```yaml
  cmd: info
  ```

### Step Options

- **`name`** - Identifier for the step (required)
- **`device`** - Device to command (required)
- **`cmd`** - Command to execute (required)
- **`params`** - Command parameters (optional)
- **`timeout`** - Max execution time in seconds (default: 10.0)
- **`expect`** - Expected value range `[min, max]` (optional)

## Advanced Features

### Simulating Realistic Device Behavior

#### Measurement Noise
```yaml
params:
  noise_std: 0.5  # Gaussian noise with std dev 0.5
```

#### Variable Latency
```yaml
params:
  latency_mean: 0.1    # Average 100ms
  latency_jitter: 0.02 # ±20ms variation
```

#### Sensor Drift
```yaml
params:
  drift_rate: 0.01  # Drifts 0.01 units per second
```

#### Fault Injection
```yaml
params:
  fail_rate: 0.2  # 20% chance of command failure
```

### Example: Advanced Recipe

See `src/config/examples/advanced_demo.yml` for a more complex example with:
- Multiple devices with different characteristics
- Drift simulation
- Unreliable device handling
- Various sensor types

## Testing

Run the test suite (requires pytest):

```bash
pip install pytest pytest-asyncio
python -m pytest tests/ -v
```

## Extending the Framework

### Adding Custom Device Types

1. Create a new class inheriting from `BaseDevice`
2. Implement `connect()`, `disconnect()`, and `command()` methods
3. Register it with the factory:

```python
from src.devices import DeviceFactory, BaseDevice

class MyDevice(BaseDevice):
    async def connect(self):
        # Connection logic
        pass
    
    async def disconnect(self):
        # Disconnection logic
        pass
    
    async def command(self, cmd, params=None):
        # Command logic
        return {"status": "ok", "value": result}

# Register the new type
DeviceFactory.register_device_type("mydevice", MyDevice)
```

## Troubleshooting

### Recipe fails to load
- Check YAML syntax
- Ensure all required fields are present (`name`, `devices`, `steps`)
- Verify device `type` is valid (currently only `simulated` is built-in)

### Step timeout
- Increase `timeout` value in step definition
- Check `latency_mean` and `latency_jitter` are reasonable

### Expected range validation fails
- Check `expect` range is appropriate for device characteristics
- Consider `noise_std` when setting tolerances
- Account for `drift_rate` in longer recipes

### Command errors
- Verify command name is correct: `set`, `read`, or `info`
- Ensure required params are provided (e.g., `set` needs `value`)
- Check device exists in `devices` section

## Next Steps

1. ✅ Run the demo recipes to understand the framework
2. ✅ Create your own simple recipe
3. ✅ Experiment with device parameters (noise, latency, drift)
4. ✅ Add fault injection to test error handling
5. ✅ Build custom device types for your use case

For more examples, see the `src/config/examples/` directory.
