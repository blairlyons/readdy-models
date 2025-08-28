# Simularium ReaDDy Models
Tools for building computational biology models and example models from the Simularium project. Includes coarse-grained monomer [ReaDDy](https://readdy.github.io/) models for actin, microtubules, and kinesin.

---

## Quickstart

See [examples/README.md](examples/README.md) to run example actin, microtubules, or kinesin models either locally or on AWS with Docker.

## Installation

**Development Head:** `pip install git+https://github.com/simularium/simularium_readdy_models.git`<br>

### Development Install with `conda`:

(`conda` is currently required to install `readdy`.)

1. Create a virtual environment with conda-specific dependencies: `conda env create -f environment.yml`
2. Activate the environment: `conda activate readdy_models`
3. Install just command runner: `conda install -c conda-forge just`
3. Install remaining dependencies: `just install`

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

