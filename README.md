# DeepMarkPy Benchmark

DeepMark Benchmark is a modular and scalable Python platform for evaluating the robustness of audio watermarking systems. It enables testing against various attacks, including both simple signal manipulations and advanced AI-based disruptions, using a containerized architecture for consistency and ease of use.

## Features

*   **Extensible Plugin System:** Easily add new watermarking models and attacks.
*   **Containerized Services:** Key models and attacks run as isolated Docker services for dependency management and reproducibility.
*   **Centralized Configuration:** Service network ports are managed via a single `.env` file.
*   **Client-Server Architecture:** The benchmark runner communicates with containerized plugins via HTTP.
*   **Standardized Execution:** Provides a CLI for running benchmarks and collecting results.

## Architecture Overview

This benchmark uses a client-server architecture. Core watermarking models and complex attacks (often AI-based) run as independent web services managed by Docker Compose. The main benchmark script (`src/run.py`) acts as a client, communicating with these services via HTTP requests over a Docker network to perform embedding, attacking, and detection. This isolates complex dependencies within containers.

## Prerequisites

*   Python 3.9+
*   Docker (Install Docker)
*   Docker Compose (Install Docker Compose)

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/deepmarkpy/deepmarkpy-benchmark.git
cd deepmarkpy-benchmark
```

### 2. Review Environment File (`.env`)

This repository includes a `.env` file which defines the default network ports used by the various Docker services (models and attacks). Docker Compose automatically reads this file when starting the services.

*   **Action:** Review the ports defined in the `.env` file. You generally don't need to change the defaults unless they conflict with other services already running on your machine. If a conflict exists, modify the corresponding port number in the `.env` file before proceeding.

### 3. Install Core Dependencies (Optional - for development/direct script interaction)

It's recommended to use a virtual environment for the benchmark runner itself:

Linux/macOS: 
```bash
python3 -m venv venv
source venv/bin/activate
```

Windows
```bash
python -m venv venv
venv\Scripts\activate
```

Install core benchmark runner dependencies:

```bash
pip install -r requirements.txt
```

### 4. Install Docker (For AI-Based Attacks and Models)
If you plan to use AI-powered attacks or models, install [Docker](https://docs.docker.com/engine/install/) and [Docker Compose](https://docs.docker.com/compose/install/).

### 5. Install Rubberband (Windows Only)

If using time stretch and pitch shift attacks on Windows, you'll need Rubberband CLI:

1. Download Rubberband CLI:
   - Get Windows executable from [Rubber Band website](https://breakfastquay.com/rubberband/)

2. Extract Files:
   - Unzip to a directory (e.g. C:\Program Files\rubberband)

3. Add to PATH:
   - Open System Properties > Advanced > Environment Variables
   - Under System Variables, find "Path"
   - Click Edit > New
   - Add your rubberband directory path
   - Click OK to save

## Running the Benchmark

### 1. Build and Start Services

This command builds the Docker images for all containerized models/attacks (defined in `docker-compose.yml`) using the configuration from `.env` and starts them in the background. This step is **required** if you intend to use plugins like `audioseal`, `vae`, `diffusion`, etc.
```bash
docker-compose up --build -d
```
You can check the status of the services using `docker-compose ps`. The first build might take some time.

### 2. Run the CLI
Ensure the Docker services are running (`docker-compose up -d`) if you are using containerized plugins. Then, execute the main benchmark script from your activated virtual environment (if used) or directly:

```bash
python src/run.py --wav_files_dir /path/to/your/audio/files/dir/ \
                  --wm_model AudioSealModel \
                  --attack_types VAEAttack SpeechEnhancementAttack \
                  # Add any other specific attack parameters like --zero_cross_pause_length 25
```

### 3. View Results

The benchmark will generate:
- benchmark_results.json – Stores detailed attack results.
- benchmark_stats.json – Summary of attack effectiveness.

## Adding a New Plugin

DeepMark Benchmark is designed to allow easy addition of new attacks and watermarking models.

### Adding a New Attack

1.	Create a New Attack Folder

Inside src/plugins/attacks, create a new folder with the attack name:
```Shell
mkdir src/plugins/attacks/new_attack
```
2.	Add attack.py
Create a file attack.py inside your folder:
```python 
import numpy as np
from core.base_attack import BaseAttack

class NewAttack(BaseAttack):
    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """Applies the attack and returns the modified audio."""
        # Example: Invert the audio signal
        return -audio
```
3.	Add config.json
```json 
{
    "attack_parameter": 0.5
}
```
4.	Dockerizing (Optional)

    If your attack requires AI models:
  - Add app.py for FastAPI service.
  - Add port to the .env file.
  - Write a Dockerfile to containerize it.
  - Add it to docker-compose.yml.

5.	Run the Benchmark
```bash 
python src/run.py --wav_files_dir path/to/audio --wm_model AudioSealModel --attack_types NewAttack
```

### Adding a New Watermarking Model

1.	Create a New Model Folder

Inside src/plugins/models, create a folder:
```Shell 
mkdir src/plugins/models/new_model
```

2.	Add model.py
```python
import numpy as np
from core.base_model import BaseModel

class NewModel(BaseModel):
    def embed(self, audio: np.ndarray, watermark_data: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Embeds a watermark in the audio."""
        return audio + 0.01 * watermark_data

    def detect(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Detects watermark from the audio."""
        return np.random.randint(0, 2, size=16)
```

3.	Add config.json
```json
{
    "watermark_size": 16
}
```

4.	Run the Benchmark with the New Model
```Shell
python src/run.py --wav_files_dir path/to/audio --wm_model NewModel --attack_types CutSamplesAttack
```

### Docker Integration

To run AI-based plugins inside Docker:
```Shell
docker-compose up --build -d
```
To stop:
```shell
docker-compose down
```

## Contributing

We welcome contributions! Feel free to:
- Report issues
- Suggest new features
- Submit pull requests

## Citation

If you use DeepMarkPy Benchmark in your research, please cite our paper:

```
@inproceedings{
kovacevic2025deepmark,
title={DeepMark Benchmark: Redefining Audio Watermarking Robustness},
author={Slavko Kova{\v{c}}evi{\'c} and Murilo Z. Silvestre and Kosta Pavlovi{\'c} and Petar Nedi{\'c} and Igor Djurovi{\'c}},
booktitle={The 1st Workshop on GenAI Watermarking},
year={2025},
url={https://openreview.net/forum?id=56ZC5dqvJO}
}
```

## License

This project is licensed under MIT License.
