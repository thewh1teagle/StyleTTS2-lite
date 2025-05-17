import torch
import librosa
import yaml
import math
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from munch import Munch

class LearnedDownSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type

        if self.layer_type == 'none':
            self.conv = nn.Identity()
        elif self.layer_type == 'timepreserve':
            self.conv = nn.Conv2d(dim_in, dim_in, kernel_size=(3, 1), stride=(2, 1), groups=dim_in, padding=(1, 0))
        elif self.layer_type == 'half':
            self.conv = nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), groups=dim_in, padding=1)
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)
            
    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == 'half':
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample='none'):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.downsample_res = LearnedDownSample(downsample, dim_in)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample_res(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class StyleEncoder(nn.Module):
    def __init__(self, dim_in=48, style_dim=48, max_conv_dim=384):
        super().__init__()
        blocks = []
        blocks += [nn.Conv2d(1, dim_in, 3, 1, 1)]

        repeat_num = 4
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 5, 1, 0)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.Linear(dim_out, style_dim)

    def forward(self, x):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        s = self.unshared(h)
    
        return s

class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.fc = nn.Linear(style_dim, channels*2)

    def forward(self, x, s):
        x = x.transpose(2, 1)
        x = x.transpose(1, 2)
                
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, 2), beta.transpose(1, 2)
        
        
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, 2).transpose(2, 1)

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, 2)


class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        self.lstm = nn.LSTM(channels, channels//2, 1, batch_first=True, bidirectional=True)

        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels),
                actv,
                nn.Dropout(0.2),
            ))
        # self.cnn = nn.Sequential(*self.cnn)
    
    def forward(self, x, input_lengths, m):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.to(input_lengths.device).unsqueeze(1)
        x.masked_fill_(m, 0.0)
        
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
            
        x = x.transpose(1, 2)  # [B, T, chn]

        input_lengths = input_lengths.cpu()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)
                
        x = x.transpose(2, 1)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

        x_pad[:, :, :x.shape[-1]] = x
        x = x_pad.to(x.device)
        
        x.masked_fill_(m, 0.0)
        
        return x

    def inference(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x
    
    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask


class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=True)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode='nearest')


class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2),
                 upsample='none', dropout_p=0.0):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)
        
        if upsample == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(nn.ConvTranspose1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1, output_padding=1))
        
        
    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class ProsodyPredictor(nn.Module):

    def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
        super().__init__() 
        
        self.add_module('text_encoder', DurationEncoder(sty_dim=style_dim, 
                                            d_model=d_hid,
                                            nlayers=nlayers, 
                                            dropout=dropout))

        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, max_dur)
        
        self.shared = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.F0 = nn.ModuleList()
        self.F0.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))

        self.N = nn.ModuleList()
        self.N.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))
        
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)


    def forward(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)
        
        # predict duration
        input_lengths = text_lengths
        x = nn.utils.rnn.pack_padded_sequence(
            d, input_lengths, batch_first=True, enforce_sorted=False)
        
        m = m.to(text_lengths.device).unsqueeze(1)
        
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)
        
        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]])

        x_pad[:, :x.shape[1], :] = x
        x = x_pad.to(x.device)
                
        duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=self.training))
        
        en = (d.transpose(2, 2) @ alignment)

        return duration.squeeze(-1), en
    
    def F0Ntrain(self, x, s):
        x, _ = self.shared(x.transpose(2, 1))
        
        F0 = x.transpose(2, 1)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)

        N = x.transpose(2, 1)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)
        
        return F0.squeeze(1), N.squeeze(1)
    
    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask


class DurationEncoder(nn.Module):

    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(nn.LSTM(d_model + sty_dim, 
                                 d_model // 2, 
                                 num_layers=1, 
                                 batch_first=True, 
                                 bidirectional=True, 
                                 dropout=dropout))
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))
        
        
        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward(self, x, style, text_lengths, m):
        masks = m.to(text_lengths.device)
        
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
                
        x = x.transpose(0, 1)
        input_lengths = text_lengths.cpu()
        x = x.transpose(2, 1)
        
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                r = block(x.transpose(2, 1), style)
                x = r.transpose(2, 1)
                x = torch.cat([x, s.permute(1, 2, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(2, 1), 0.0)
            else:
                x = x.transpose(2, 1)
                x = nn.utils.rnn.pack_padded_sequence(
                    x, input_lengths, batch_first=True, enforce_sorted=False)
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(
                    x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x.transpose(2, 1)
                
                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

                x_pad[:, :, :x.shape[-1]] = x
                x = x_pad.to(x.device)
        return x.transpose(2, 1)
    
    def inference(self, x, style):
        x = self.embedding(x.transpose(2, 1)) * math.sqrt(self.d_model)
        style = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, style], axis=-1)
        src = self.pos_encoder(x)
        output = self.transformer_encoder(src).transpose(0, 1)
        return output
    
    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

class TextCleaner:
    def __init__(self, symbol_dict, debug=True):
        self.word_index_dictionary = symbol_dict
        self.debug = debug
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError as e:
                if self.debug:
                    print("\nWARNING UNKNOWN IPA CHARACTERS/LETTERS: ", char)
                    print("To ignore set 'debug' to false in the config")
                continue
        return indexes

class Preprocess():
    def __init__(self, config_path, models_path, device=torch.device('cpu')):
        self.device = device
        config = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
        params = config['model_params']
        try:
            symbols = (
                            list(config['symbol']['pad']) +
                            list(config['symbol']['punctuation']) +
                            list(config['symbol']['letters']) +
                            list(config['symbol']['letters_ipa']) +
                            list(config['symbol']['extend'])
                        )
            symbol_dict = {}
            for i in range(len((symbols))):
                symbol_dict[symbols[i]] = i
            
            n_token = len(symbol_dict) + 1
            print("\nFound:", n_token, "symbols")
        except Exception as e:
            print(f"\nERROR: Cannot find {e} in config file!\nYour config file is likely outdated, please download updated version from the repository.")
            raise SystemExit(1)
        
        self.cleaner       = TextCleaner(symbol_dict, debug=False)
        self.style_encoder = StyleEncoder(dim_in=params['dim_in'], 
                                          style_dim=params['style_dim'], 
                                          max_conv_dim=params['hidden_dim']).to(device)# acoustic style encoder
        self.__load_models(models_path)
    def __load_models(self, models_path):
        model = {'style_encoder':self.style_encoder}

        params_whole = torch.load(models_path, map_location='cpu')
        params = params_whole['net']
        params = {key: value for key, value in params.items() if key in model.keys()}

        for key in model:
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

    def __wave_preprocess(self, wave):
        to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        mean, std = -4, 4
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
        return mel_tensor
    
    def __length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask    
    
    def preprocess_input(self, phonem, speaker):
        #preprocess text
        tokens = self.cleaner(phonem)
        tokens.insert(0, 0)
        tokens.append(0)
        tokens = torch.LongTensor(tokens).unsqueeze(0)
        text_mask = self.__length_to_mask(torch.LongTensor([tokens.shape[-1]]))
        #preprocess audio
        wave, sr = librosa.load(speaker['path'], sr=24000)
        audio, _ = librosa.effects.trim(wave, top_db=30)
        if sr != 24000: audio = librosa.resample(audio, sr, 24000)

        mel = self.__wave_preprocess(audio).unsqueeze(1)

        return tokens, text_mask, mel, speaker['speed']
    
    def get_style(self, mel):
        return self.style_encoder(mel.to(self.device))

#For ONNX inference only
class StyleTTS2(torch.nn.Module):
    def __init__(self, config_path, models_path, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.ref_s = None
        config = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
        
        try:
            symbols = (
                            list(config['symbol']['pad']) +
                            list(config['symbol']['punctuation']) +
                            list(config['symbol']['letters']) +
                            list(config['symbol']['letters_ipa']) +
                            list(config['symbol']['extend'])
                        )
            symbol_dict = {}
            for i in range(len((symbols))):
                symbol_dict[symbols[i]] = i
            
            n_token = len(symbol_dict) + 1
        except Exception as e:
            print(f"\nERROR: Cannot find {e} in config file!\nYour config file is likely outdated, please download updated version from the repository.")
            raise SystemExit(1)
        
        args = self.__recursive_munch(config['model_params'])
        args['n_token'] = n_token

        assert args.decoder.type in ['istftnet', 'hifigan'], 'Decoder type unknown'
    
        if args.decoder.type == "istftnet":
            from Modules.ONNX.hifigan import Decoder
            self.decoder = Decoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                    resblock_kernel_sizes = args.decoder.resblock_kernel_sizes,
                    upsample_rates = args.decoder.upsample_rates,
                    upsample_initial_channel=args.decoder.upsample_initial_channel,
                    resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                    upsample_kernel_sizes=args.decoder.upsample_kernel_sizes, 
                    gen_istft_n_fft=args.decoder.gen_istft_n_fft, gen_istft_hop_size=args.decoder.gen_istft_hop_size) 
        elif args.decoder.type == "hifigan":
            from Modules.ONNX.hifigan import Decoder
            self.decoder = Decoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                    resblock_kernel_sizes = args.decoder.resblock_kernel_sizes,
                    upsample_rates = args.decoder.upsample_rates,
                    upsample_initial_channel=args.decoder.upsample_initial_channel,
                    resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                    upsample_kernel_sizes=args.decoder.upsample_kernel_sizes) 
            
        self.predictor           = ProsodyPredictor(style_dim=args.style_dim, d_hid=args.hidden_dim, nlayers=args.n_layer, max_dur=args.max_dur, dropout=args.dropout)
        self.text_encoder        = TextEncoder(channels=args.hidden_dim, kernel_size=5, depth=args.n_layer, n_symbols=args.n_token)

        self.__load_models(models_path)
        self.eval()
        self.to(self.device)
    
    def __onnx_alignment(self, pred_dur, max_length):
        seq_length = pred_dur.sum().to(dtype=torch.int64)
        device = pred_dur.device
                
        clipped_dur = torch.clamp(pred_dur, 0, seq_length)
        
        zero_tensor = torch.zeros(1, device=device)
        prefix_sum = torch.cat([zero_tensor, torch.cumsum(clipped_dur[:-1], dim=0)])
        
        positions = torch.arange(seq_length, device=device).unsqueeze(0).expand(max_length, -1)
        
        start_positions = prefix_sum.unsqueeze(1)
        
        end_positions = torch.min(
            start_positions + clipped_dur.unsqueeze(1),
            start_positions.new_full(start_positions.shape, seq_length, dtype=torch.float)
        )
        
        mask = (positions >= start_positions) & (positions < end_positions)

        return mask.float()
    
    def __recursive_munch(self, d):
        if isinstance(d, dict):
            return Munch((k, self.__recursive_munch(v)) for k, v in d.items())
        elif isinstance(d, list):
            return [self.__recursive_munch(v) for v in d]
        else:
            return d
        
    def __replace_outliers_zscore(self, tensor, threshold=3.0, factor=0.95):
        mean = tensor.mean()
        std = tensor.std()
        z = (tensor - mean) / std

        # Identify outliers
        outlier_mask = torch.abs(z) > threshold
        # Compute replacement value, respecting sign
        sign = torch.sign(tensor - mean)
        replacement = mean + sign * (threshold * std * factor)

        result = tensor.clone()
        result[outlier_mask] = replacement[outlier_mask]

        return result
        
    def __load_models(self, models_path):
        model = {'decoder':self.decoder, 'predictor':self.predictor, 'text_encoder':self.text_encoder}

        params_whole = torch.load(models_path, map_location='cpu')
        params = params_whole['net']
        params = {key: value for key, value in params.items() if key in model.keys()}

        for key in model:
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
        
    def forward(self, tokens, text_mask, style, speed=1):
        device = self.device
        tokens = tokens.to(device)
        
        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)

            # encode
            t_en = self.text_encoder(tokens, input_lengths, text_mask)
            s = style.to(device)
        
            # cal alignment
            d = self.predictor.text_encoder(t_en, s, input_lengths, text_mask)
            x, _ = self.predictor.lstm(d)
            duration = self.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)

            duration[:,1:-2] = self.__replace_outliers_zscore(duration[:,1:-2]) #Normalize outlier
            
            duration /= speed

            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            if torch.onnx.is_in_onnx_export():
                pred_aln_trg = self.__onnx_alignment(pred_dur, input_lengths[0])
            else:
                pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
                c_frame = 0
                for i in range(pred_aln_trg.size(0)):
                    pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                    c_frame += int(pred_dur[i].data)
            alignment = pred_aln_trg.unsqueeze(0).to(device)

            # encode prosody
            en = (d.transpose(2, 1) @ alignment)
            F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
            asr = (t_en @ alignment)

            out = self.decoder(asr, F0_pred, N_pred, s)
            out = out.squeeze()[4000:-4000] #Remove weird pulse and silent tokens
        return out