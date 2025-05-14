"""
uv run Demo/infer.py
"""
#import libs
import soundfile as sf
import os
import sys
import torch
import traceback
import random
import numpy as np
from pathlib import Path
device = 'cuda' if torch.cuda.is_available() else 'cpu' #setup GPU

#import espeak
#If you did not use eSpeak for your language, please add your own G2P.
import sys
import phonemizer
if sys.platform.startswith("win"):
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        import espeakng_loader
        EspeakWrapper.set_library(espeakng_loader.get_library_path())
    except Exception as e:
        print(e)

#import inference
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from inference import StyleTTS2
#########################################CHANGE YOUR PATH HERE#########################################
config_path = str(Path("Configs") / "config.yaml")
models_path = str(Path('Models') / 'Finetune/base_model.pth')
#######################################################################################################
voice_path = str(Path("Demo") / "Audio")
model = StyleTTS2(config_path, models_path).eval().to(device)

def main():
    phonemes = 'ðə mjuːzˈiəm ɡˈɑːɹd nˈɛvɚɹ ɛkspˈɛktᵻd ðə skˈʌlptʃɚ tə mˈuːv, bˌʌt æt pɹɪsˈaɪsli mˈɪdnaɪt, ɪts ˈaɪz blˈɪŋkt, ænd ɪts lˈɪps kˈɜːld ˌɪntʊ ɐ nˈoʊɪŋ smˈaɪl, æz ɪf ɐwˈeɪkənɪŋ fɹʌm sˈɛntʃɚɹiz ʌv sˈaɪləns.'
    speed = 1
    denoise = 0.2
    avg_style = True
    stabilize = True

    speaker = {
        "path": 'Demo/Audio/7_alloy.wav',
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