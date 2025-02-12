

# DeepMark Benchmark

DeepMark Benchmark is a modular and scalable platform for evaluating the robustness of audio watermarking systems. It enables testing against various attacks, including both simple signal manipulations and advanced AI-based disruptions.
 
---

## Installation

### 1. Clone the Repository
```Shell
git clone https://github.com/your-repo/deepmark-benchmark.git
cd deepmark-benchmark
```
### 2. Create and Activate a Virtual Environment
Linux/Mac:
```Shell
python3 -m venv venv
source venv/bin/activate
```
Windows:
```Shell
python -m venv venv
venv\Scripts\activate
```
### 3. Install Dependencies
```Shell
pip install -r requirements.txt
```
### 4. Install Docker (For AI-Based Attacks and Models)
If you plan to use AI-powered attacks or models, install [Docker](https://docs.docker.com/engine/install/) and [Docker Compose](https://docs.docker.com/compose/install/).

## Running the Benchmark
### 1. Ensure Docker Containers Are Running
For AI-based attacks and models, start the services:
```Shell
docker-compose up --build -d
```
### 2. Run the CLI
```Shell
python src/run.py --wav_files_dir path/to/audio --model AudioSealModel --attacks CutSamplesAttack TimeStretchAttack --num_flips 100 --flip_duration 0.5
```
- --wav_files_dir: Path to the directory containing .wav files.
- --model: The watermarking model to use (AudioSealModel, WavMarkModel, etc.).
- --attacks: List of attacks to apply (CutSamplesAttack, PitchShiftAttack, etc.).

Additional parameters depend on the attack configurations.

### 3. View Results
The benchmark will generate:
- benchmark_results.json – Stores detailed attack results.
- benchmark_stats.json – Summary of attack effectiveness.

---


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
  - Add server.py for FastAPI service.
  - Write a Dockerfile to containerize it.
  - Add it to docker-compose.yml.
5.	Run the Benchmark
```Shell 
python src/run.py --wav_files_dir path/to/audio --model AudioSealModel --attacks NewAttack
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
python src/run.py --wav_files_dir path/to/audio --model NewModel --attacks CutSamplesAttack
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
<hr/>
## Contributing
We welcome contributions! Feel free to:
- Report issues
- Suggest new features
- Submit pull requests
  
---

## License
This project is licensed under MIT License.
