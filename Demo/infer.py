"""
wget https://huggingface.co/dangtr0408/StyleTTS2-lite/resolve/main/Models/base_model.pth -O Models/base_model.pth
wget https://huggingface.co/dangtr0408/StyleTTS2-lite/resolve/main/Models/config.yaml -O Configs/config.yaml

uv sync --extra demo
uv run Demo/infer.py
"""
import soundfile as sf
import sys
import torch
import numpy as np
from pathlib import Path


root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from inference import StyleTTS2



def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #setup GPU
    config_path = str(Path("Configs") / "config.yaml")
    models_path = str(Path('Models') / 'Finetune/base_model.pth')
    model = StyleTTS2(config_path, models_path).eval().to(device)
    
    phonemes = 'həlˈO wˈɜɹld!'
    speed = 1
    denoise = 0.2
    avg_style = True
    stabilize = True

    speaker = {
        "path": 'Demo/Audio/10_michael.wav',
        "speed": speed
    }

    with torch.no_grad():
        styles = model.get_styles(speaker, denoise, avg_style)
        r = model.generate(phonemes, styles, stabilize, 18)
        r = r / np.max(np.abs(r))  # Normalize

    sr = 24000
    sf.write("audio.wav", r, sr)
    print('Created audio.wav')
    
main()