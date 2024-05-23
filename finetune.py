from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from transformers import AutoFeatureExtractor, WhisperModel
from transformers import LlamaTokenizer
from datasets import load_dataset
import torch, torchaudio
from torch import nn
import numpy as np
from jiwer import wer as calculate_wer
import pickle
import fire
from datasets import Dataset, Audio, Value
import os, random
from typing import Optional
from whisper.normalizers import EnglishTextNormalizer
import math
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
from pathlib import Path
import whisper
import copy, heapq
normalizer = EnglishTextNormalizer()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train(
    MODEL = "openai/whisper-large-v3",
    DATASET = "chime4",
    TRAIN_DATA = "",
    DEV_DATA = "",
    SAVE_EVERY = 10,
    BATCH_SIZE = 32,
    GRADIENT_ACCUMULATION_STEPS = 4,
    LEARNING_RATE = 1e-3,
    EPOCHS = 100,
    THRESOLD=2.0,
    TOP_PERCENT=0.8,
    TAU=10,
    ):
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL)
    processor = WhisperProcessor.from_pretrained(MODEL, language="en", task="transcribe")
    tokenizer = WhisperTokenizer.from_pretrained(MODEL, language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL).to(device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    state_dict = copy.deepcopy(model.state_dict())

    prompt_and_eos = tokenizer('')['input_ids']
    prompt_ids, eos_id = prompt_and_eos[:-1], prompt_and_eos[-1]

    def data_preparation(data_path, feature_extractor, tokenizer):
        with open(data_path + "wav.scp", 'r') as f1:
            wave_data = f1.readlines()
        with open(data_path + "text", 'r') as f2:
            trans_data = f2.readlines()

        audio_data, txt_data = [], []
        for i in range(len(wave_data)):
            audio_data.append(wave_data[i])
            txt_data.append(trans_data[i])

        audio_dataset = []
        all_pred, all_gt = [], []
        for audio_line, text_line in zip(audio_data, txt_data):
            audio_path = audio_line.strip().split()[1]
            text = ' '.join(text_line.split()[1:]).lower().strip()
            audio, sr = torchaudio.load(audio_path)
            if sr != 16000:
                audio = torchaudio.functional.resample(audio, sr, 16000)
            item = {'audio': audio, 'text': text}

            item['mel'] = feature_extractor(audio.squeeze(0).numpy(), sampling_rate=16_000, return_tensors="pt")['input_features']
            item['decoder_input_ids'] = tokenizer(text, max_length=1024, truncation=True).input_ids

            model.load_state_dict(state_dict)
            hidden_feature = model.model.encoder(input_features=item['mel'].to(device)).last_hidden_state

            # prompt: '<|startoftranscript|><|en|><|transcribe|><|notimestamps|>'
            pseudo_label_ids = torch.tensor([prompt_ids]).long().to(device)

            ### probs: confidence score
            probs, decoder_outputs = [], None
            for _ in range(150):
                decoder_outputs = model(encoder_outputs=(hidden_feature), decoder_input_ids=pseudo_label_ids, output_attentions=True)
                logits = torch.softmax(decoder_outputs.logits / 1.2, dim=-1)
                next_token = logits[0, -1, :].topk(1)[1]
                probs.append(float(logits[0, -1, next_token]))
                pseudo_label_ids = torch.cat((pseudo_label_ids, next_token.unsqueeze(0)), dim=-1)
                if next_token == eos_id:     # EOS
                    break
            
            # normlization
            mean_probs = sum(probs) / len(probs)
            for k in range(len(probs)):
                probs[k] = round(probs[k] / mean_probs, 3)

            ### weights: attentive score
            n_prompt_toks = 4
            layer_id, head_id = 30, 13  # suggest: layer_id \in [30,31], head_id \in [0,1,...,19]
            attn = decoder_outputs.decoder_attentions[layer_id][0, head_id, :, :]
            attn[:, :n_prompt_toks-1] = 0   # remove prompts
            weights = []
            for i in range(n_prompt_toks-1, attn.shape[-1]):
                weight = torch.sum(attn[i, :]) + torch.sum(attn[:, i]) - attn[i, i]
                weights.append(float(weight))
            
            # normalization
            mean_weights = sum(weights) / len(weights)
            for j in range(len(weights)):
                weights[j] = round(weights[j] / mean_weights, 3)

            ### final_weights: star score
            final_weights = []
            for ci, ai in zip(probs, weights):
                c_over_a, a_over_c = ci * ci / ai, ai * ai / ci
                conflict = (sigmoid((c_over_a - THRESOLD) * TAU) + sigmoid((a_over_c - THRESOLD) * TAU)) * ai
                no_conflict = (sigmoid((THRESOLD - c_over_a) * TAU) * sigmoid((THRESOLD - a_over_c) * TAU)) * ai * np.exp((ci - ai) / TAU)
                final_weights.append(conflict + no_conflict)

            item['pseudo_label_ids'] = pseudo_label_ids
            item['probs'] = torch.tensor(final_weights).unsqueeze(0)
            pseudo_text = processor.batch_decode(pseudo_label_ids, skip_special_tokens=True)[0]

            ### utt-level uncertainty
            if 'train' in data_path:
                avg_wer, generated_texts = 0, []
                for _ in range(5):
                    new_state_dict = copy.deepcopy(state_dict)
                    for k in new_state_dict.keys():
                        std = torch.std(new_state_dict[k])
                        noise = torch.randn_like(new_state_dict[k])
                        new_state_dict[k] = new_state_dict[k] + noise * std * 0.1

                    model.load_state_dict(new_state_dict)
                    generated_ids = model.generate(inputs=item['mel'].to(device), forced_decoder_ids=forced_decoder_ids, max_new_tokens=150)
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    generated_texts.append(generated_text)
                    avg_wer += calculate_wer([pseudo_text], [generated_text]) / 5

                item['avg_wer'] = avg_wer
                item['diversity'] = len(list(set(generated_texts)))

            ## text normalization
            pseudo_text = normalizer(pseudo_text)
            pseudo_text = pseudo_text if len(pseudo_text) > 0 else '<UNK>'

            gt = normalizer(text)
            gt = gt if len(gt) > 0 else '<UNK>'

            audio_dataset.append(item)
            all_pred.append(pseudo_text)
            all_gt.append(gt)

        model.load_state_dict(state_dict)
        return audio_dataset, calculate_wer(all_gt, all_pred)


    def evaluate(model, dataset):
        with torch.no_grad():
            all_pred, all_gt = [], []
            for item in dataset:
                mel = item['mel']
                generated_ids = model.generate(inputs=mel.to(device), forced_decoder_ids=forced_decoder_ids, max_new_tokens=150)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                ## text normalization
                pred = normalizer(generated_text)
                pred = pred if len(pred) > 0 else '<UNK>'

                gt = normalizer(item['text'])
                gt = gt if len(gt) > 0 else '<UNK>'

                all_pred.append(pred)
                all_gt.append(gt)

        return calculate_wer(all_gt, all_pred)


    model.eval()
    train_dataset, train_wer = data_preparation(TRAIN_DATA, feature_extractor, tokenizer)
    dev_dataset, dev_wer = data_preparation(DEV_DATA, feature_extractor, tokenizer)
    os.system('mkdir -p data')
    torch.save(train_dataset, f'data/train_{DATASET}.pt')
    torch.save(dev_dataset, f'data/dev_{DATASET}.pt')
    model.train()

    ## load saved data
    # train_dataset = torch.load(f'data/train_{DATASET}.pt')
    # dev_dataset = torch.load(f'data/dev_{DATASET}.pt')

    ## utt-level filtering
    def product(item):
        return item['avg_wer'] * item['diversity']
    filtered_train_dataset = heapq.nsmallest(int(len(train_dataset) * TOP_PERCENT), train_dataset, key=product)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    model_size = MODEL.replace('openai/whisper-', '')
    exp_dir = f'runs/{DATASET}_{model_size}'
    os.system(f"mkdir -p {exp_dir}")

    steps, loss = 0, 0
    best_loss, best_wer = 10000, 10000
    for Epoch in range(EPOCHS):
        print("Epoch: ", Epoch + 1)

        # Train
        random.shuffle(filtered_train_dataset)
        print('Training...')
        for i in range(len(filtered_train_dataset) // BATCH_SIZE):
            batch_data = filtered_train_dataset[i * BATCH_SIZE: (i+1) * BATCH_SIZE]

            input_features = [{"input_features": item["mel"]} for item in batch_data]
            mel = processor.feature_extractor.pad(input_features, return_tensors="pt")["input_features"].squeeze(1).to(device)

            labels = batch_data[0]["pseudo_label_ids"].to(device)
            y_in = labels[:, :-1]
            y_out = labels[:, 1:]

            logits = model(input_features=mel, decoder_input_ids=y_in).logits
            loss_items = loss_fn(logits.permute(0, 2, 1), y_out)

            # uncertainty calibration
            ratios = batch_data[0]['probs'].to(device)
            ratios = ratios / torch.mean(ratios)
            loss = (torch.sum(loss_items[:, :n_prompt_toks-1]) + torch.sum(loss_items[:, n_prompt_toks-1:] * ratios)) / (n_prompt_toks-1 + ratios.shape[-1])

            loss.backward()
            steps += 1

            if steps % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if steps % SAVE_EVERY == 0:   # Evaluate
                torch.save(model, f"{exp_dir}/Iter_{steps}.pth")

                model.eval()
                dev_wer = evaluate(model, dev_dataset)
                model.train()

                if dev_wer < best_wer or (dev_wer == best_wer and loss < best_loss):
                    torch.save(model, f"{exp_dir}/best_checkpoint.pth")
                    best_loss, best_wer = loss, dev_wer

    torch.save(model, f"{exp_dir}/last_checkpoint.pth")


if __name__ == "__main__":
    fire.Fire(train)

