"""
uv sync
wget https://huggingface.co/dangtr0408/StyleTTS2-lite/resolve/main/Models/base_model.pth
wget https://huggingface.co/dangtr0408/StyleTTS2-lite/resolve/main/Models/config.yaml

uv pip install onnx espeakng-loader phonemizer-fork onnxruntime
uv run ONNX/export_onnx.py
"""
import torch
from inference_onnx import StyleTTS2, Preprocess
import sys
import onnxruntime
from pathlib import Path
import phonemizer
import soundfile as sf

if sys.platform in ['win32', 'darwin']:
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    import espeakng_loader
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

def get_phoneme(text, lang):
    try:
        my_phonemizer = phonemizer.backend.EspeakBackend(language=lang, preserve_punctuation=True,  with_stress=True, language_switch='remove-flags')
        return my_phonemizer.phonemize([text])[0]
    except Exception as e:
        print(e)

#Model path
config_path = str(Path("Configs") / "config.yaml")
models_path = str(Path('Models') / 'Finetune/base_model.pth')

#Inputs
speaker = {
    "path": "Demo/Audio/1_heart.wav",             #Ref audio path
    "speed": torch.tensor(1.0, dtype=torch.float32),#Speaking speed
}
text = 'Nearly 300 scholars currently working in the United States have applied for positions at Aix Marseille University in France.'

def main():
    with torch.no_grad():
        preprocess = Preprocess(config_path, models_path)

        phonemes = get_phoneme(text=text, lang="en-us")
        tokens, mel, speed = preprocess.preprocess_input(phonemes, speaker)
        style              = preprocess.get_style(mel)
        model_inputs = {
            "tokens": tokens.numpy(),
            "style": style.numpy(),
            "speed": speed.numpy(),
        }
        
        sess = onnxruntime.InferenceSession("model.onnx")
        wav = sess.run(None, model_inputs)[0]

        sf.write('audio.wav', wav, 24000)
        print('Created audio.wav')

    

if __name__ == "__main__":
    main()

