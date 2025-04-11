import torch
# torch.manual_seed(0)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

# import random
# random.seed(0)

# import numpy as np
# np.random.seed(0)

# load packages
import yaml
import numpy as np
import torch
import torchaudio
import librosa
from nltk.tokenize import word_tokenize
import unicodedata
import noisereduce as nr
import re

from models import *
from utils import *
from text_utils import TextCleaner

import phonemizer
from phonemizer.backend.espeak.wrapper import EspeakWrapper 
EspeakWrapper.set_library('C:\Program Files\eSpeak NG\libespeak-ng.dll')

def espeak_phn(text, lang):
    my_phonemizer = phonemizer.backend.EspeakBackend(language=lang, preserve_punctuation=True,  with_stress=True, language_switch='remove-flags')
    return my_phonemizer.phonemize([text])[0]

def load_models(config_path, models_path, device):
    config = yaml.safe_load(open(config_path))
    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params)

    keys_to_keep = {'predictor', 'decoder', 'text_encoder','predictor_encoder', 'style_encoder'}

    params_whole = torch.load(models_path, map_location='cpu')
    params = params_whole['net']
    params = {key: value for key, value in params.items() if key in keys_to_keep}

    for key in list(model.keys()):
        if key not in keys_to_keep:
            del model[key]

    for key in model:
        if key in params:
            try:
                model[key].load_state_dict(params[key])
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                model[key].load_state_dict(new_state_dict, strict=False)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]

    module_params = []
    for module_name in model:
        total_params = sum(p.numel() for p in model[module_name].parameters())
        print(module_name,":",total_params)
        module_params.append(total_params)
    print('\nTotal',":",sum(module_params))

    return model

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
    mean, std = -4, 4
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def text_normalize(text):
    punctuation = ["，", "、", "،", ";", "(", "．", "。", "…", "!", "–", ":"]
    map_to = "."
    punctuation_pattern = re.compile(f"[{''.join(re.escape(p) for p in punctuation)}]")
    #ensure consistency.
    text = unicodedata.normalize('NFKC', text)
    #replace punctuation that acts like a comma or period
    #text = re.sub(r'\.{2,}', '.', text)
    text = punctuation_pattern.sub(map_to, text)
    #remove or replace special chars except . , { } ? ' -  \ 
    text = re.sub(r'[^\w\s.,{}?\'\-\[\]]', ' ', text)
    #replace consecutive whitespace chars with a single space and strip leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def merge_fragments(texts, n):
    merged = []
    i = 0
    while i < len(texts):
        fragment = texts[i]
        j = i + 1
        while len(fragment.split()) < n and j < len(texts):
            fragment += ", " + texts[j]
            j += 1
        merged.append(fragment)
        i = j
    if len(merged[-1].split()) < n and len(merged) > 1: #handle last sentence
        merged[-2] = merged[-2] + ", " + merged[-1]
        del merged[-1]
    else:
        merged[-1] = merged[-1]
    return merged

def text_preprocess(text, n_merge=12):
    text_norm = text_normalize(text).replace(",", ".").split(".")#split.
    text_norm = [s.strip() for s in text_norm]
    text_norm = list(filter(lambda x: x != '', text_norm)) #filter empty index
    text_norm = merge_fragments(text_norm, n=n_merge) #merge if a sentence has less that n 
    return text_norm

def init_replacement_func(replacements):
    replacement_iter = iter(replacements)
    def replacement(match):
        return next(replacement_iter)
    return replacement


def compute_style(model, path, denoise=False, split_dur=0):
    if split_dur != 0: split_dur = max(int(split_dur), 1)
    device = next(model.decoder.parameters()).device
    wave, sr = librosa.load(path, sr=24000)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
        sr = 24000

    max_samples = sr*15 #max 15 seconds ref audio
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    if denoise:
        audio = nr.reduce_noise(y=audio, sr=sr, n_fft=2048, win_length=1200, hop_length=300)

    with torch.no_grad():
        if split_dur>0 and len(audio)/sr>split_dur:
            count = 0
            ref_s = None
            jump = sr*split_dur
            total_len = len(audio)
            
            #Need to init before the loop
            mel_tensor = preprocess(audio[0:jump]).to(device)
            ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))
            count += 1
            for i in range(jump, total_len, jump):
                if i+jump >= total_len:
                    left_dur = (total_len-i)/sr
                    if left_dur >= 0.5: #Still count if left over dur is >= 0.5s
                        mel_tensor = preprocess(audio[i:total_len]).to(device)
                        ref_s += model.style_encoder(mel_tensor.unsqueeze(1))
                        ref_p += model.predictor_encoder(mel_tensor.unsqueeze(1))
                        count += 1
                    continue
                mel_tensor = preprocess(audio[i:i+jump]).to(device)
                ref_s += model.style_encoder(mel_tensor.unsqueeze(1))
                ref_p += model.predictor_encoder(mel_tensor.unsqueeze(1))
                count += 1
            ref_s /= count
            ref_p /= count
        else:
                mel_tensor = preprocess(audio).to(device)
                ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
                ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1)

def inference(model, phonem, ref_s, speed=1, prev_d_mean=0, t1=0.1, t2=0.1):
    device = next(model.decoder.parameters()).device
    speed = min(max(speed-1, -1), 1) #speed range [0, 2]
    
    phonem = ' '.join(word_tokenize(phonem))
    tokens = TextCleaner()(phonem)
    tokens.insert(0, 0)
    tokens.append(0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        # encode
        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        s = ref_s[:, :128]
        s_dur = ref_s[:, 128:]
        s_dur = s_dur*(1-t1) + torch.empty(s_dur.shape).normal_(mean=s_dur.mean(),std=s_dur.std()).to(device)*t1 #Add a bit of randomness to the voice's tone
    
        # cal alignment
        d = model.predictor.text_encoder(t_en, s_dur, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) - (2 * speed)

        if prev_d_mean != 0:
            exclude = [duration[:,0].clone(), duration[:,-1].clone()]

            dur_stats = torch.empty(duration.shape).normal_(mean=prev_d_mean, std=duration.std()).to(device)
            duration = duration*(1-t2) + dur_stats*t2

            duration[:,0] = exclude[0]
            duration[:,1] = exclude[1]
            
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)
        alignment = pred_aln_trg.unsqueeze(0).to(device)

        # encode prosody
        en = (d.transpose(-1, -2) @ alignment)
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s_dur)
        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))

        out = model.decoder(asr, F0_pred, N_pred, s)
        
    return out.squeeze().cpu().numpy(), duration.mean()


def generate(model, text, speakers, ref_s_speakers, default_speaker="[id_1]", n_merge=16, randomness=0.2, smooth_dur=0.2):
    list_wav        = []
    prev_d_mean     = 0
    lang_pattern    = r'\[([^\]]+)\]\{([^}]+)\}'

    text = re.sub(r'[\n\r\t\f\v]', '', text)
    #fix lang tokens span to multiple sents
    find_lang_tokens = re.findall(lang_pattern, text)
    if find_lang_tokens:
        cus_text = []
        for lang, t in find_lang_tokens:
            parts = text_preprocess(t, n_merge=0)
            parts = ".".join([f"[{lang}]" + f"{{{p}}}"for p in parts])
            cus_text.append(parts)
        replacement_func = init_replacement_func(cus_text)
        text = re.sub(lang_pattern, replacement_func, text)

    texts = re.split(r'(\[id_\d+\])', text) #split the text by speaker ids while keeping the ids.
    if len(texts) <= 1:
        texts.insert(0, default_speaker)
    texts = list(filter(lambda x: x != '', texts))

    for i in texts:
        if bool(re.match(r'(\[id_\d+\])', i)):
            #Set up env for matched speaker
            speaker_id = i.strip('[]')
            current_ref_s = ref_s_speakers[speaker_id]
            speed = speakers[speaker_id]['speed']
            continue
        text_norm = text_preprocess(i, n_merge=n_merge)
        for sentence in text_norm:
            cus_phonem = []
            find_lang_tokens = re.findall(lang_pattern, sentence)
            if find_lang_tokens:
                for lang, t in find_lang_tokens:
                    try:
                        phonem = espeak_phn(t, lang)
                        cus_phonem.append(phonem)
                    except Exception as e:
                        print(e)
                    
            replacement_func = init_replacement_func(cus_phonem)
            phonem =  espeak_phn(sentence, speakers[speaker_id]['lang'])
            phonem = re.sub(lang_pattern, replacement_func, phonem)

            wav, prev_d_mean = inference(model, phonem, current_ref_s, speed=speed, prev_d_mean=prev_d_mean, t1=randomness, t2=smooth_dur)
            wav = wav[7000:-7000] #Remove weird pulse and silent tokens
            list_wav.append(wav)
    
    final_wav = np.concatenate(list_wav)
    final_wav = np.concatenate([np.zeros([12000]), final_wav, np.zeros([12000])], axis=0) # 0.5 second padding
    return final_wav