# StyleTTS2-lite

## Base Model
_If you have a better checkpoint, your contribution would be greatly appreciated!_

**English**:
- English checkpoint (LibriTTS 100,000 steps): [Download (MIT)](https://huggingface.co/dangtr0408/StyleTTS2-lite/tree/main/Models)  
- LibriTTS dataset (CC BY 4.0): [Download](https://huggingface.co/datasets/dangtr0408/LibriTTS-clean-460/tree/main) 
- Demo: [StyleTTS2-lite-space](https://huggingface.co/spaces/dangtr0408/StyleTTS2-lite-space)

**Vietnamese**:
- Vietnamese checkpoint (viVoice 120,000 steps): [Download (CC BY 4.0 SA)](https://huggingface.co/dangtr0408/StyleTTS2-lite-vi/tree/main/Models)
- viVoice dataset (CC BY 4.0 SA): [Download](https://huggingface.co/datasets/capleaf/viVoice) 
- Demo: [StyleTTS2-lite-vi-space](https://huggingface.co/spaces/dangtr0408/StyleTTS2-lite-vi-space)

Model Component Parameter Summary

| Component              | Parameters   | Used in Inference |
|------------------------|--------------|--------------------|
| **Decoder**            | 54,289,492   | ✅ Yes             |
| **Predictor**          | 16,194,612   | ✅ Yes             |
| **Style Encoder**      | 13,845,440   | ✅ Yes             |
| **Text Encoder**       | 5,606,400    | ✅ Yes             |
| **Text Aligner (ASR)** | 7,865,252    | ❌ No              |
| **Pitch Extractor (JDC)** | 5,248,067  | ❌ No              |
| **mpd (Discriminator)**| 41,105,770   | ❌ No              |
| **msd (Discriminator)**| 280,902      | ❌ No              |
| **Total**              | **144,435,935** |                |


## How To Start Finetuning

**1. Install the requirements**
```bash
pip  install  -r  requirements.txt
```

**2. Download model and config file**

[Download](https://huggingface.co/dangtr0408/StyleTTS2-lite/tree/main)
and place the base model in ***/Models/Finetune*** and the corresponding config file in ***/Configs***.

**3. Format your data like StyleTTS2, but exclude the speaker field.**

Format: filename.wav | transcription

For reference, see ***val.txt*** in [LibriTTS dataset](https://huggingface.co/datasets/dangtr0408/LibriTTS-clean-460/tree/main).

**4. (Optional) Extend the token set**

If you want to add new tokens to train on another language:
- Make sure that you did step 2.
- Open extend.ipynb, edit the number of tokens you would like to extend to and run the notebook.
- Find the new weights and configs in ***/Extend/New_Weights/***, then replace the originals.
- Add your new IPA symbols to the _extend list in meldataset.py ⚠️ Ensure the total number of symbols in meldataset.py matches your extended token count!
- (Note) If you are using my **inference.py from StyleTTS2-lite-vi on HuggingFace** make sure to update it with new IPA list as well.

**5. Adjust your configs file**

For a single GPU with 24 GB VRAM, i find the following works well. Some users have reported that setting max_len below 300 can slightly degrade quality. Personally, I haven’t encountered any issues even with max_len 180.
<pre lang="yaml">
batch_size: 2 
max_len: 310 # maximum number of frames
</pre>

Change the location of your dataset. For example:
<pre lang="yaml">
data_params:
  train_data: ../Data_Speech/LibriTTS/train.txt
  val_data: ../Data_Speech/LibriTTS/val.txt
  root_path: ../Data_Speech/
</pre>

**6. Start training**
```bash
python train.py
```

## Disclaimer  

**Before using these pre-trained models, you agree to inform the listeners that the speech samples are synthesized by the pre-trained models, unless you have the permission to use the voice you synthesize. That is, you agree to only use voices whose speakers grant the permission to have their voice cloned, either directly or by license before making synthesized voices public, or you have to publicly announce that these voices are synthesized if you do not have the permission to use these voices.**


## References

- [yl4579/StyleTTS2](https://arxiv.org/abs/2306.07691)

- [jik876/hifi-gan](https://github.com/jik876/hifi-gan)

## License

**Code: MIT License**