# Polymer Informatics Pipeline

Automated simulation and ML pipeline for polymer capacitance prediction.

## Quick Start

### Prerequisites
- Python 3.10+
- Ansys 2024 (for simulation only - optional for data processing)
- 4GB RAM minimum
- Git

## Important: Ansys Student Edition Limitations ⚠️

If using Ansys Student Edition, be aware of these critical limitations:

### Hardware/Software Limits
- **Maximum model size**: 32,000 nodes per simulation
- **Maximum iterations**: 10,000
- **Parallel processing**: ❌ NOT SUPPORTED (single-threaded only)
- **License duration**: Expires after 12 months

### Software/Commercial Limits
- **Usage**: Educational only (non-commercial)
- **Publications**: Cannot publish research results
- **Commercial deployment**: ❌ NOT PERMITTED
- **Support**: Limited to university email addresses

### Pipeline Implications
```text
Your Setup: Ansys Student Edition
├─ max_parallel_jobs: MUST be 1 (not 4)
├─ Processing time: 4-8x slower than commercial
├─ Dataset size limit: ~200-500 samples (before exceeding node limit)
└─ Recommendation: Upgrade to commercial for production
```

### Checking Your Ansys Version
```bash
# Check if Student Edition
ansys2024 --version | grep -i student

# Recommended upgrade path
# 1. Academic: Use university license
# 2. Research: Apply for student research extension
# 3. Commercial: Purchase full license
```

### For Production Deployment
⚠️ **You CANNOT deploy this pipeline commercially with Student Ansys**

Options:
1. Upgrade to Commercial Ansys
2. Use institutional Ansys license
3. Deploy ML pipeline only (without simulation)

### Installation

#### Option 1: Local Development Setup
```bash
# Clone repository
git clone https://github.com/zapzapzap12v/Polymer_Informatics.git
cd Polymer_Informatics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development

# Install pre-commit hooks
pre-commit install

# Run tests to verify installation
pytest tests/
```

#### Option 2: Docker
```bash
# Build Docker image
docker build -t polymer_informatics:latest .

# Run container
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  polymer_informatics:latest --help
```

#### Option 3: Docker Compose (with services)
```bash
docker-compose up
```

## Configuration

### Environment Variables

Create `.env` file or export these:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANSYS_EXECUTABLE` | auto-detect | Path to Ansys binary |
| `ANSYS_LICENSE_SERVER` | localhost | Ansys license server |
| `ANSYS_LICENSE_PORT` | 1055 | License server port |
| `CONFIG_FILE` | config/default_config.yaml | Configuration file path |
| `LOG_LEVEL` | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `MAX_PARALLEL_JOBS` | 4 | Maximum parallel Ansys simulations |
| `RANDOM_SEED` | 42 | Random seed for reproducibility |

### Configuration File

See `config/default_config.yaml` for all options:

```yaml
ansys:
  version: "2024"
  license_server: ansyslicense.company.com
  license_port: 1055
  max_parallel_jobs: 4
  job_timeout_minutes: 60

paths:
  working_directory: ./work
  temp_directory: ~/.polymerinfo/temp
  output_directory: ./results
  log_directory: ./logs
  data_directory: ./data

simulation:
  max_retries: 3
  retry_backoff_seconds: 5

dataset_management:
  min_samples_per_material_class: 100
  stratified_cv_required: true

logging:
  level_file: DEBUG
  level_console: INFO
  enable_rotation: true
  file_max_bytes: 10485760
```

## Usage

### Basic Pipeline Run

```bash
python main.py
```

With custom config:
```bash
python main.py --config custom_config.yaml
```

With specific dataset:
```bash
python main.py --input-data samples.csv --output results.csv
```

### Data Processing Only

```bash
python main.py --mode=data-processing
```

### ML Training Only

```bash
python main.py --mode=training --trained-model model.pkl
```

### Validation Only

```bash
python main.py --mode=validation
```

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_config_manager.py

# With coverage
pytest --cov=. --cov-report=html

# Specific test marker
pytest -m "not slow"

# Integration tests
pytest tests/integration_tests/ -v
```

### Code Quality Checks

```bash
# Format with black
black .

# Lint with flake8
flake8 .

# Type check with mypy
mypy .

# Check import order
isort --check-only .

# Run all checks (pre-commit)
pre-commit run --all-files
```

## Project Structure

```
Polymer_Informatics/
├── config/                      # Configuration management
│   ├── config_manager.py
│   ├── default_config.yaml
│   └── examples/
├── simulation/                  # Ansys simulation interface
│   ├── simulation_manager.py
│   ├── adaptive_retry_manager.py
│   └── ansys_interface.py
├── analysis/                    # Data analysis tools
│   ├── target_analyzer.py
│   └── dataset_composition_manager.py
├── data/                        # Data processing pipeline
│   ├── data_loader.py
│   ├── data_processor.py
│   ├── data_validation.py
│   └── streaming_result_manager.py
├── models/                      # ML models and training
│   ├── training/
│   │   ├── model_trainer.py
│   │   └── reproducibility.py
│   └── validation/
│       └── cross_validation.py
├── utils/                       # Utility functions
│   ├── input_validation.py
│   ├── logging_setup.py
│   ├── type_validation.py
│   ├── numerical_stability.py
│   └── encoding_utils.py
├── tests/                       # Test suite
│   ├── test_*.py
│   ├── integration_tests/
│   └── fixtures/
├── main.py                      # Entry point
├── requirements.txt             # Dependencies
├── Dockerfile                   # Container image
├── docker-compose.yml           # Multi-container setup
├── README.md                    # This file
├── CHANGELOG.md                 # Version history
└── .env.example                 # Environment variable template
```

## Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     POLYMER INFORMATICS PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT DATA
   │
   ├─→ [Data Loader]
   │        │
   │        ├─→ CSV Validation
   │        ├─→ Encoding Detection
   │        └─→ Schema Validation
   │
   ├─→ [Data Processor]
   │        │
   │        ├─→ Null Handling
   │        ├─→ Type Conversion
   │        ├─→ Bounds Checking
   │        └─→ Normalization
   │
   ├─→ [Data Validator]
   │        │
   │        └─→ Quality Metrics
   │
   ├─→ [Composition Manager]
   │        │
   │        ├─→ Material Class Analysis
   │        ├─→ Stratification Check
   │        └─→ Distribution Validation
   │
   ├─→ [Streaming Result Manager]
   │        │
   │        └─→ Batch Writing (Memory Efficient)
   │
   ├─→ [Target Analyzer]        ← DECISION POINT
   │        │                      Is target achievable?
   │        ├─→ Distribution Analysis
   │        ├─→ Physics Validation
   │        ├─→ Viable Input Space
   │        └─→ Gap Analysis
   │
   ├─→ [Model Training]
   │        │
   │        ├─→ Reproducibility (Set Seed)
   │        ├─→ Cross-Validation
   │        │    ├─→ Per-Material-Class Metrics
   │        │    ├─→ Error Analysis
   │        │    └─→ Statistical Testing
   │        └─→ Checkpoint Saving
   │
   └─→ [Output Results]
        │
        ├─→ CSV Results
        ├─→ Model Pickle
        ├─→ Metrics Report
        └─→ Logs

OPTIONAL: Ansys Simulation Loop (if needed)
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SIMULATION PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [Parameters] → [Retry Manager] → [Simulation Diagnostics] → [Results]    │
│                      │                                                      │
│                      ├─ Attempt 1: Default settings                        │
│                      ├─ Attempt 2: Remesh adaptive                         │
│                      ├─ Attempt 3: Solver adjustment                       │
│                      └─ Attempt 4+: Geometry healing                       │
│                                                                             │
│  Success Rate: 38% → 75% (after adaptive retry)                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Performance & Scalability

### Single Machine (4 CPU cores)
- 1440 samples: ~8 hours (with Ansys simulation)
- 7200 samples: ~40 hours

### HPC Cluster (128 CPU cores)
- 1440 samples: ~30 minutes
- 7200 samples: ~2.5 hours

### Cloud (AWS EC2)
- Use t3.xlarge (4 cores) + EBS for data: ~$1 per hour
- Use c5.9xlarge (36 cores) for faster processing

## Troubleshooting

### Ansys Not Found
```bash
# Check if Ansys is installed
which ansys  # Linux
where ansys  # Windows

# Set explicit path
export ANSYS_EXECUTABLE=/path/to/ansys

# Or in config.yaml
ansys:
  executable_path: /path/to/ansys
```

### License Server Unreachable
```bash
# Test connection
telnet ansyslicense.company.com 1055

# Set correct server in config
ansys:
  license_server: ansyslicense.company.com
  license_port: 1055
```

### Memory Exhaustion
```bash
# Use streaming result manager (automatic in production)
# Or reduce batch size
simulation:
  batch_size: 100
```

### Tests Failing
```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Check Python version
python --version  # Must be 3.10+

# Run with verbose output
pytest -v -s
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes and commit: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature/my-feature`
5. Submit pull request

### Development Setup
```bash
pip install -r requirements-dev.txt
pre-commit install
pytest tests/
```

## Testing Strategy

### Unit Tests (`tests/test_*.py`)
Fast, isolated tests of individual components.
Target: >80% code coverage

### Integration Tests (`tests/integration_tests/`)
End-to-end tests of data pipelines.
Target: All critical workflows covered

### Performance Tests (`tests/test_performance.py`)
Ensure performance meets SLOs.
Target: Process 5000 samples/second

## Production Deployment

### Pre-Deployment Checklist

- [ ] All tests passing: `pytest tests/`
- [ ] Code style approved: `black . && flake8 .`
- [ ] Type hints complete: `mypy .`
- [ ] Docker image builds: `docker build -t polymer_informatics:latest .`
- [ ] Docker image runs: `docker run --rm polymer_informatics:latest --help`
- [ ] Integration test passes: `pytest tests/integration_tests/`
- [ ] ⚠️ **CRITICAL - Ansys License Check:**
  - [ ] Confirm Ansys edition: Commercial or Student?
  - [ ] If Student: Confirm non-commercial use only
  - [ ] If Commercial: Document license agreement
  - [ ] Test max_parallel_jobs setting
  - [ ] Verify node limit won't be exceeded
- [ ] Configuration validated: `python config/config_manager.py`
- [ ] Reproducibility confirmed: seed test passes
- [ ] Documentation updated: `CHANGELOG.md` reflects changes
- [ ] Version bumped: `setup.py` has new version

### Ansys Student Edition Deployment Restrictions
⚠️ **CANNOT DEPLOY IF:**
- Using Student Ansys for commercial purposes
- Expecting to process >500 samples (risk of exceeding 32K node limit)
- Requiring parallel processing (max_parallel_jobs > 1)
- Planning to publish results commercially

✅ **CAN DEPLOY IF:**
- Using commercial Ansys license
- Using Student Ansys for educational/internal research only
- Have upgraded to full license
- Using ML pipeline only (no simulation)

### Ansys Commercial License Pre-Requisites
- [ ] Valid commercial Ansys license file/server configured
- [ ] License server reachable: `telnet $ANSYS_LICENSE_SERVER $ANSYS_LICENSE_PORT`
- [ ] Multiple parallel jobs tested successfully
- [ ] Large geometry processing tested (verify node count)

### Deployment Steps

```bash
# 1. Verify configuration
python -c "from codes.ansys_license_validator import AnsysLicenseValidator; \
           from codes.config_manager import ConfigurationLoader; \
           config = ConfigurationLoader.load_configuration(); \
           valid, msg = AnsysLicenseValidator.enforce_student_limits(config); \
           print(msg)"

# 2. Run integration tests
python tests/run_integration_test.py

# 3. Build Docker image (for ML pipeline)
docker build -t polymer_informatics:production .

# 4. Deploy
docker run -d \
  --name polymer_informatics \
  -v /data:/app/data \
  -v /results:/app/results \
  polymer_informatics:production
```

### Post-Deployment

- [ ] Health checks passing
- [ ] Logs show no errors
- [ ] Processing throughput meets SLOs
- [ ] Alert monitoring configured

## Performance Monitoring

### Key Metrics

- Simulation success rate (target: >75%)
- Target achievement rate (target: >20%)
- Average processing time per sample
- Memory usage
- Disk usage

### Logging

All logs written to:
- File: `logs/pipeline.log` (rotated daily)
- Console: INFO level and above

Set log level in config:
```yaml
logging:
  level_console: DEBUG  # For development
  level_file: DEBUG
```

## Known Limitations

1. **Ansys Simulation**: Requires Windows/Linux with Ansys license
2. **Scalability**: Limited by Ansys licensing (typically 4-8 parallel jobs)
3. **Material Classes**: Only polymers, alloys, nanocomposites currently supported
4. **Frequency Range**: 1 Hz - 1 GHz only

## Future Enhancements

- [ ] Support for composite materials
- [ ] Multi-frequency analysis
- [ ] GPU acceleration for ML models
- [ ] Cloud-native deployment (Kubernetes)
- [ ] REST API interface
- [ ] Web dashboard for results

## Citation

If you use this pipeline in research, please cite:
```bibtex
@software{polymer_informatics_2024,
  author = {Your Team},
  title = {Polymer Informatics: Automated Simulation and ML Pipeline},
  year = {2024},
  url = {https://github.com/zapzapzap12v/Polymer_Informatics}
}
```

## License

MIT License - see LICENSE file

## Support

- Issues: GitHub Issues
- Discussions: GitHub Discussions
- Email: your-email@company.com

## Changelog

See `CHANGELOG.md` for version history.
