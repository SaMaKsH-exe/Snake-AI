# Snake-AI

Minimal Deep Q-Learning project for Snake.

## Setup
1. Create a Conda environment:
   ```bash
   conda create -p .conda python=3.11 -y
   ```

2. Install dependencies:
   ```bash
   conda install --prefix .conda pytorch torchvision torchaudio cpuonly -c pytorch -y
   pip install matplotlib numpy IPython
   ```

## Run
Run the training script:
```bash
python agent.py
```

## Notes
- To speed up training, you can increase the `SPEED` value in `snake.py` (e.g., to 1000), but this will make the game run too fast to follow visually.


I followed the tutorial: https://www.youtube.com/watch?v=L8ypSXwyBds
