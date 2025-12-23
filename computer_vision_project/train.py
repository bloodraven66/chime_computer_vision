import utils
import os
import json
from tqdm import tqdm
import torch
import cv2
import numpy as np
import torchaudio
import jiwer, math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

os.environ["HF_HOME"] = "/mnt/matylda4/udupa/huggingface"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)        
        pe = pe.unsqueeze(0)        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

def args():
    cfg = {}
    ##dataset related
    cfg["train_path"] = "/mnt/matylda2/data/MCoRec/data-bin/train/"
    cfg["dev_path"] = "/mnt/matylda2/data/MCoRec/data-bin/dev/"
    cfg["audio_save_folder"] = "/mnt/matylda4/udupa/data/dump/chime2026/computer_vision_project/audio/##MODE##/##SESSION##.wav"
    cfg["metadata_file_name"] = "metadata.json"
    cfg["audio_sample_rate"] = 16000
    cfg["label_folder"] = "labels"
    cfg["min_duration"] = 1.0  # in seconds
    cfg["max_duration"] = 10.0  # in seconds

    ##features
    cfg["audio_features"] = "melspec"
    cfg["lips"] = "raw"

    ##model related
    cfg["lip_encoder"] = "2d_conv"
    cfg["audio_encoder"] = "lstm"
    cfg["merger"] = "cross_attention"
    cfg["decoder"] = "ctc"

    ##run and model related
    cfg["batch_size"] = 8
    cfg["num_epochs"] = 20
    cfg["learning_rate"] = 1e-4
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg

class Model(torch.nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        if "decoder" in cfg:
            if cfg["decoder"] == "ctc":
                self.decoder = torch.nn.Linear(512, cfg["num_tokens"])
        if "lip_encoder" in cfg:
            if cfg["lip_encoder"] == "2d_conv":
                
                self.lip_encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2),
                    torch.nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2),
                    torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Flatten(start_dim=1),
                    torch.nn.Linear(256 * 3 * 3, 512),  # Changed from 4*4 to 3*3
                    torch.nn.ReLU()
                )

                self.lip_temporal_encoder = torch.nn.LSTM(
                    512, 256, batch_first=True, bidirectional=True, dropout=0.1,
                )
                
            self.lip_norm = torch.nn.LayerNorm(512)
        if "audio_encoder" in cfg:
            if cfg["audio_encoder"] == "lstm":
                # self.audio_subsampler = torch.nn.Sequential(
                #     torch.nn.Conv1d(104, 104, kernel_size=3, stride=2, padding=1), # 100Hz -> 50Hz
                #     torch.nn.ReLU(),
                #     torch.nn.Conv1d(104,104, kernel_size=3, stride=2, padding=1), # 50Hz -> 25Hz
                #     torch.nn.ReLU()
                # )

                self.audio_encoder = torch.nn.LSTM(
                    input_size=104,
                    hidden_size=256,
                    num_layers=3,
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.1
                )
            self.audio_norm = torch.nn.LayerNorm(512)

        if "merger" in cfg:
            if cfg["merger"] == "cross_attention":
                self.merger = torch.nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True, dropout=0.1)
                self.pos_embed = PositionalEncoding(d_model=512)

                self.refiner = torch.nn.LSTM(
                input_size=512, hidden_size=256, num_layers=3,
                batch_first=True, bidirectional=True, dropout=0.1
            )
        
    def forward(self, lip_feats, audio_feats, lip_lengths, audio_lengths):
        #24, 104, 1, 88, 88]) torch.Size([24, 104, 104
        # print("in forward", lip_feats.shape, audio_feats.shape)
        lip_feats = lip_feats.float() 
        audio_feats = audio_feats.float()
        # Encode lip features
        if hasattr(self, 'lip_encoder'):
            if self.cfg["lip_encoder"] == "2d_conv":
                B, T, C, H, W = lip_feats.size()
                lip_feats = lip_feats.view(B * T, C, H, W)
                lip_encoded = self.lip_encoder(lip_feats)
                lip_encoded = lip_encoded.view(B, T, -1)  # (B, T, D) #32, 155, 512
            lip_encoded, _ = self.lip_temporal_encoder(lip_encoded)
            lip_encoded = self.lip_norm(lip_encoded)
            lip_encoded = self.pos_embed(lip_encoded)
        # Encode audio features
        if hasattr(self, 'audio_encoder'):
            if self.cfg["audio_encoder"] == "lstm":
                # audio_encoded, _ = self.audio_encoder(audio_feats)  # (B, T, D) #32, 616, 512
                # print(audio_feats.shape, audio_lengths)
                # audio_feats = self.audio_subsampler(audio_feats.transpose(1, 2)).transpose(1, 2)  # (B, T, Mel) #32, 154, 80
                # print(audio_feats.shape, audio_lengths)
                packed_audio = pack_padded_sequence(
                audio_feats, 
                audio_lengths.cpu(), 
                batch_first=True, 
                enforce_sorted=False
            )
                packed_output, _ = self.audio_encoder(packed_audio)
                audio_encoded, _ = pad_packed_sequence(
                    packed_output, 
                    batch_first=True
                )
            audio_encoded = self.audio_norm(audio_encoded)
            audio_encoded = self.pos_embed(audio_encoded)
        lip_mask = self._create_padding_mask(lip_lengths, lip_encoded.size(1))
        # Mask for audio features: (B, T_audio)
        audio_mask = self._create_padding_mask(audio_lengths, audio_encoded.size(1))

        # Merge features
        if hasattr(self, 'refiner'):
            if self.cfg["merger"] == "cross_attention":
                # merged_feats, _ = self.merger(audio_encoded, lip_encoded, lip_encoded)  # (B, T, D)
                merged_feats, _ = self.merger(
                    audio_encoded,      # query
                    lip_encoded,        # key
                    lip_encoded,        # value
                    key_padding_mask=lip_mask  # Mask out lip padding
                )
                merged_feats = merged_feats + audio_encoded  # Residual connection
                # merged_feats =  audio_encoded + lip_encoded
                merged_feats, _ = self.refiner(merged_feats)

                
        # Decode
        if hasattr(self, 'decoder'):
            if self.cfg["decoder"] == "ctc":
                outputs = self.decoder(merged_feats)  # (B, T, Vocab)
                outputs = outputs.log_softmax(dim=-1)
        return outputs

    def _create_padding_mask(self, lengths, max_len):
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=lengths.device)[None, :] >= lengths[:, None]
        return mask

    def loss(self, outputs, targets, input_lengths, target_lengths):
        if self.cfg["decoder"] == "ctc":
            outputs = outputs.transpose(0, 1)  # (B,T,V) -> (T,B,V)
            return torch.nn.functional.ctc_loss(
                outputs, targets, input_lengths, target_lengths,
                blank=1, zero_infinity=True
            )

def trainer(model, optimizer, loader, mode="train", id2char=None):
    text_transform = utils.TextTransform(
        sp_model_path="../mcorec_baseline/src/tokenizer/spm/unigram/unigram5000.model",
        dict_path="../mcorec_baseline/src/tokenizer/spm/unigram/unigram5000_units.txt"
    )
    if mode != "train":
        beam_search = utils.get_beam_search_decoder(model.avsr, text_transform.token_list, beam_size=1, ctc_weight=1, diar_weight=0)

    if mode == "train":
        model.train()
    else:
        preds, refs = [], []
        model.eval()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"{mode}ing")
    batch_idx = 0
    for batch in pbar:
        batch_idx += 1
        lip_feats, lip_lengths, audio_feats, audio_lengths, texts, text_lengths, raw_texts, ids = batch
        lip_feats = lip_feats.to(cfg["device"])
        audio_feats = audio_feats.to(cfg["device"])
        texts = texts.to(cfg["device"])
        audio_lengths = audio_lengths.to(cfg["device"])
        # audio_lengths = torch.div(audio_lengths, 4, rounding_mode='floor')
        # videos, 
        # audios, 
        # labels,
        # video_lengths, 
        # audio_lengths, 
        # label_lengths

        text_lengths = text_lengths.to(cfg["device"])
        lip_lengths = lip_lengths.to(cfg["device"])
        audio_feats = audio_feats.permute(0, 2, 1)  # (B, Mel, T) -> (B, T, Mel)
        lip_feats = lip_feats.permute(0, 2, 1, 3, 4)
        
        # outputs = model.forward(lip_feats, audio_feats, lip_lengths, audio_lengths)
        # loss = model.loss(outputs, texts, audio_lengths, text_lengths)
        
        if mode == "train":
            outputs = model(lip_feats, audio_feats, texts, lip_lengths, audio_lengths, text_lengths)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()        
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (pbar.n + 1):.4f}'
            })
        if mode != "train":
            avhubert_features = model.avsr.encoder(
                input_features=audio_feats, 
                video=lip_feats,
            )
            audiovisual_feat = avhubert_features.last_hidden_state
            audiovisual_feat = audiovisual_feat.squeeze(0)

            nbest_hyps = beam_search(audiovisual_feat, diar_scores=None)
            nbest_hyps = [h.asdict() for h in nbest_hyps[:min(len(nbest_hyps), 1)]]
            predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
            predicted = text_transform.post_process(predicted_token_id).replace("<eos>", "")
            print(predicted)
            # exit()
            preds.extend(predicted)
            refs.extend(raw_texts)
        if batch_idx > 10:
            break
    avg_loss = total_loss / len(loader)
    print(f"{mode.capitalize()} Loss: {avg_loss:.4f}")
    # if mode != "train":
    #     wer = jiwer.wer(refs, preds)
    #     cer = jiwer.cer(refs, preds)
    #     for i in range(3):
    #         print(f"Ref: {refs[i]}")
    #         print(f"Hyp: {preds[i]}")
    #     print(f"{mode.capitalize()} WER: {wer:.4f}, CER: {cer:.4f}")
    return avg_loss

def tokenizers(charset, blank_token_id=1, pad_id=0):
    char2id = {ch: idx + 2 for idx, ch in enumerate(charset)}
    char2id["<pad>"] = pad_id
    char2id["<blank>"] = blank_token_id
    id2char = {idx: ch for ch, idx in char2id.items()}
    return char2id, id2char

class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, data, mode="train", char2id=None, id2char=None, rate_ratio=640):
        self.cfg = cfg
        self.data = data
        self.mode = mode
        self.char2id = char2id
        self.id2char = id2char
        ##extract lip fps
        # cap = cv2.VideoCapture(data[0]['lip_path'])
        # self.fps = cap.get(cv2.CAP_PROP_FPS)
        # self.sr = cfg["audio_sample_rate"]
        # cap.release()
        self.audio_transform = utils.AudioTransform(subset=mode)
        self.video_transform = utils.VideoTransform(subset=mode)
        self.rate_ratio = rate_ratio  # audio sr / lip fps
        self.text_transform = utils.TextTransform(
            sp_model_path="../mcorec_baseline/src/tokenizer/spm/unigram/unigram5000.model",
            dict_path="../mcorec_baseline/src/tokenizer/spm/unigram/unigram5000_units.txt",
        )
    def __len__(self):
        return len(self.data)

    def extract_frames(self, lip_path, start_time, end_time):
        cap = cv2.VideoCapture(lip_path)
        frames = []
        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            current_frame += 1
        cap.release()
        if len(frames) > 0:
            frames = np.stack(frames, axis=0) 
        else:
            raise ValueError(f"No frames extracted from {lip_path} between {start_time} and {end_time}")
        return frames

    def extract_audio(self, audio_path, start_time, end_time):
        try:
            waveform, _ = torchaudio.load(audio_path, frame_offset=int(start_time * self.sr), num_frames=int((end_time - start_time) * self.sr))

        except Exception as e:
            print(f"Error extracting audio from {audio_path}: {e}, start_time: {start_time}, end_time: {end_time}")
            return None

        return waveform

    def get_lip_feats(self, frames):
        if self.cfg["lips"] == "raw":
            frames = frames.astype(np.float32) / 255.0  # Normalize to [0, 1]
            frames = np.transpose(frames, (0, 3, 1, 2))  # (Time, Channels, Height, Width)
            return frames
    
    def get_audio_feats(self, waveform):
        if self.cfg["audio_features"] == "melspec":
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sr,
                n_mels=80,
                n_fft=400,
                hop_length=160,
                win_length=400
            )(waveform)
            log_mel = torch.log(mel_spectrogram + 1e-9)
            return log_mel.squeeze(0).transpose(0, 1)  # (Time, Mel)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        ##extract lip frames
        # lip_frames = self.extract_frames(item['lip_path'], item['start_time'], item['end_time'])
        # audio_waveform = self.extract_audio(item['audio_path'], item['start_time'], item['end_time'])
        # lip_feats = self.get_lip_feats(lip_frames)
        # audio_feats = self.get_audio_feats(audio_waveform)
        raw_text = item['text']
        # text = [self.char2id[ch] for ch in raw_text if ch in self.char2id]
        text = self.text_transform.tokenize(raw_text)
        video = utils.load_video(item['lip_path'], item['start_time'], item['end_time'])
        audio = utils.load_audio(item['audio_path'], item['start_time'], item['end_time'])
        audio = utils.cut_or_pad(audio, len(video) * self.rate_ratio)

        lip_feats = self.video_transform(video)
        audio_feats = self.audio_transform(audio)
        # print(lip_feats.shape, audio_feats.shape)
        # exit()
        #(73, 1, 88, 88), (73, 104)
        return lip_feats, audio_feats, torch.tensor(text), raw_text, item['id']

class Collate:
    def __init__(self, cfg, mode="train"):
        self.cfg = cfg
        self.mode = mode

    def __call__(self, batch):
        lip_feats = [item[0] for item in batch]
        audio_feats = [item[1] for item in batch]
        texts = [item[2] for item in batch]
        raw_texts = [item[3] for item in batch]
        ids = [item[4] for item in batch]

        # Pad lip features
        lip_lengths = [feat.shape[0] for feat in lip_feats]
        max_lip_length = max(lip_lengths)
        C, H, W = lip_feats[0].shape[1], lip_feats[0].shape[2], lip_feats[0].shape[3]
        padded_lip_feats = torch.zeros(len(batch), max_lip_length, C, H, W)
        for i, feat in enumerate(lip_feats):
            padded_lip_feats[i, :feat.shape[0]] = torch.as_tensor(feat)

        # Pad audio features
        audio_lengths = [feat.shape[0] for feat in audio_feats]
        max_audio_length = max(audio_lengths)
        padded_audio_feats = torch.zeros(len(batch), max_audio_length, audio_feats[0].shape[1])
        for i, feat in enumerate(audio_feats):
            padded_audio_feats[i, :feat.shape[0], :] = torch.as_tensor(feat)

        # Pad texts
        text_lengths = [len(t) for t in texts]
        max_text_length = max(text_lengths)
        padded_texts = torch.zeros(len(batch), max_text_length, dtype=torch.long)
        for i, t in enumerate(texts):
            padded_texts[i, :len(t)] = torch.as_tensor(t)
        
        # print(padded_lip_feats.shape, padded_audio_feats.shape, padded_texts.shape)
        # 24, 126, 1, 88, 88]) torch.Size([24, 126, 104]) torch.Size([24, 94]
        return padded_lip_feats, torch.tensor(lip_lengths), padded_audio_feats, torch.tensor(audio_lengths), padded_texts, torch.tensor(text_lengths), raw_texts, ids   

def get_loader(cfg, mode="train", char2id=None, id2char=None, exclude_sessions=0, include_sessions=0):
    total_duration = 0.0
    skipped  = 0
    total_labels = []
    session_idx = 0
    true_mode = mode
    if mode == "train_dev":
        mode = "train"
    for session in tqdm(os.listdir(cfg[f"{mode}_path"]), desc=f"Processing {mode} sessions"):
        session_idx += 1
        if session_idx <= exclude_sessions and true_mode == "train":
            continue
        if session_idx > include_sessions and true_mode == "train_dev":
            continue 
        session_path = os.path.join(cfg[f"{mode}_path"], session)
        session_metadata_path = os.path.join(session_path, cfg["metadata_file_name"])
        metadata = utils.load_json(session_metadata_path)
        for speaker_id in metadata:
            speaker_data = metadata[speaker_id]
            if len(speaker_data['central']['crops']) > 1:
                continue
            for idx, track in enumerate(speaker_data['central']['crops']):
                video_path = os.path.join(session_path, track["lip"])
                audio_save_path = cfg["audio_save_folder"].replace("##MODE##", mode).replace("##SESSION##", session)
                if not os.path.exists(os.path.dirname(audio_save_path)):
                    os.makedirs(os.path.dirname(audio_save_path), exist_ok=True)
                if not os.path.exists(audio_save_path):
                    utils.extract_audio_from_mp4(video_path, audio_save_path)
                info = torchaudio.info(audio_save_path)
                audio_duration = info.num_frames / info.sample_rate

                crop_metadata_path = track["crop_metadata"]
                lip_path = os.path.join(session_path, track["lip"])
                crop_metadata = utils.load_json(os.path.join(session_path, crop_metadata_path))
                label_path = os.path.join(session_path, cfg["label_folder"], f"{speaker_id}.vtt")
                label_data, durations = utils.load_vtt(label_path, cfg)
                valid_labels = []
                for label in label_data:
                    if label['start_time'] < audio_duration and label['end_time'] <= audio_duration:
                        valid_labels.append({
                            **label, 
                            "session": session, 
                            "speaker_id": speaker_id, 
                            "video_path": video_path, 
                            "audio_path": audio_save_path, 
                            "crop_metadata": crop_metadata, 
                            "lip_path": lip_path
                        })
                    else:
                        skipped += 1
                
                total_labels.extend(valid_labels)
                total_duration += durations
    if true_mode == "train":
        charset = set()
        for label in total_labels:
            charset.update(set(label['text']))
        charset = sorted(list(charset))
        char2id, id2char = tokenizers(charset)

    total_duration = round(total_duration / 3600, 2)
    print(f"Total {mode} duration: {total_duration} hr, Total labels: {len(total_labels)}")
    print(f"Total {mode} skipped: {skipped}")
    dataset = Dataset(cfg, total_labels, mode=mode, char2id=char2id, id2char=id2char)
    collate_fn = Collate(cfg, mode=mode)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["batch_size"] if true_mode=="train" else 1,
        shuffle=(true_mode=="train"),
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True
    )
    if true_mode == "train":
        return loader, char2id, id2char
    return loader



if __name__ == "__main__":
    cfg = args()
    train_loader, char2id, id2char = get_loader(cfg, mode="train", exclude_sessions=10)
    dev_loader = get_loader(cfg, mode="train_dev", char2id=char2id, id2char=id2char, include_sessions=10)   
    cfg["num_tokens"] = len(char2id)
    eval_loader = get_loader(cfg, mode="dev", char2id=char2id, id2char=id2char)
    # model = Model(cfg)
    model_path = "../mcorec_baseline/model-bin/avsr_cocktail"
    from avhubert import AVHubertAVSR
    model = AVHubertAVSR.from_pretrained(model_path)
    print(model)
    print(f"Train loader: {len(train_loader.dataset)} samples") 
    print(f"Dev loader: {len(dev_loader.dataset)} samples")
    print(f"Eval loader: {len(eval_loader.dataset)} samples")
    model.to(cfg["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model Parameters: {num_params:.2f}M")
    for epoch in range(cfg["num_epochs"]):
        print(f"Epoch {epoch + 1}/{cfg['num_epochs']}")
        trainer(model, optimizer, train_loader, mode="train")
        trainer(model, optimizer, dev_loader, mode="dev", id2char=id2char)

    
