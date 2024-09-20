##########################
# 2 models' conversation #
##########################

# 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from faster_whisper import WhisperModel
import torch
import numpy as np
import random
import time
import soundfile as sf
import os
import json
import logging
from streaming_hubert import StreamingHubertEncoder, ApplyKmeans
import librosa
import ast
from vocoder.models import CodeGenerator
from vocoder.utils import AttrDict
import yaml
import argparse
from openai import OpenAI
from Unit2Mel.model.model import Model
from Unit2Mel.model.vocoder.vocoder import Vocoder
from Unit2Mel.utils import scan_checkpoint, load_checkpoint
from Unit2Mel.infer import std_normal, sampling_given_noise_schedule_ddim, compute_hyperparams_given_schedule, get_eval_noise_schedule
from librosa.util import normalize
from openai_tts import OpenAITTSAPI

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
MAX_WAV_VALUE = 32768.0


def asr_unit_pipeline(config, ASR, audio_path):
    """
    Transcribes an audio file using an ASR model.
    Input: 
        asr_model: model, ASR model
        audio_path: str, path to the audio file
    Output:
        transcription: str, transcribed text
    """
    if config['asr']['unit'] == True:
        TPS = 50/config['asr']['downsample']
        encoder = StreamingHubertEncoder()
        apply_kmeans = ApplyKmeans(config['asr']['km_model'], use_gpu=True)

    def transcribe(audio_path):
        # Read audio
        audio, sr = sf.read(audio_path)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        segments, info = ASR.transcribe(audio, beam_size=5, language=config['asr']['lan'], condition_on_previous_text=False, word_timestamps=True)
        return segments

    def quantize(audio_path, downsample):
        feat = encoder.encode(audio_path)
        ssl_units = apply_kmeans(feat)
        return [f"<|{p}|>" for p in ssl_units][::downsample]

    def combine(kms, segments):
        words = []
        for segment in segments:
            for w in segment.words:
                words.append((w.word, int(w.start * TPS)))
        for i, (w, s) in enumerate(words):
            # print(w, s)
            kms.insert(i + s, ' ' + w)

        return ''.join(kms), ' '.join(w for w, _ in words)

    # Transcribe given audio
    segments = transcribe(audio_path)

    if config['asr']['unit'] == False:
        text = ' '.join([w.word for s in segments for w in s.words])
        return text, text, text

    # Quantize Causal HuBERT features
    kms = quantize(audio_path, config['asr']['downsample'])

    # Generate interleaving sequence
    interleave, text = combine(kms, segments)
    
    return interleave, kms, text

def openai_llm_pipeline(openai_model, prompt):
    """
    Generates text using an OpenAI language model.
    Input:
        openai_model: model, OpenAI language model
        prompt: str, prompt text
    Output:
        generated_text: str, generated text
    """
    logger.info(f"Generating text from prompt: {prompt}")
    response = openai_model.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5
    )
    generated_text = response.choices[0].text.strip()
    logger.info(f"Generated text: {generated_text}")
    return generated_text

def slm_pipeline(config, slm_model, conversation_history, model_name, device):
    """
    Generates text using a SLM model.
    Input:
        slm_model: model, SLM model
        tokenizer: tokenizer, tokenizer
        conversation_history: list, conversation history
        model_name: str, name of the model
    Output:
        generated_text: str, generated text
    """
    logger.debug(f"Model {model_name} prompts: {conversation_history}")
    if config["mode"] == "cascade":
        client = OpenAI()
        response = client.chat.completions.create(
            model=config["slm"]["model_name"],
            messages=conversation_history,
            )
        text = response.choices[0].message.content
        return text, text, text
    else:
        slm_model, tokenizer = slm_model
    inputs = tokenizer.apply_chat_template(
        conversation_history,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(device)
    with torch.no_grad():
        outputs = slm_model.generate(
            inputs,
            temperature=1.0,
            max_new_tokens=1024
        )
    logger.debug(f"Model {model_name} generated output: {outputs}")
    generated_ids = outputs[0][inputs.size(1):]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    logger.info(f"Model {model_name} generated text: {generated_text}")
    def _post_process(result_str: str):
        results = result_str.replace('|>', '|> ').replace('<|', ' <|').replace('  ', ' ').split(' ')
        kms = []
        words = []
        for u in results:
            if u == "<|eot_id|>":
                break
            if u[:2] == '<|' and u[-2:] == '|>':
                kms.append(u[2:-2])
            else:
                words.append(u)
        return kms, words
    
    tokens, text = _post_process(generated_text)
    return generated_text, tokens, ' '.join(text)



# For E2E_Spoken_LM, https://github.com/YK-Fu/E2E_Spoken_LM/tree/main
def tts_pipeline(config, tts_model, data, output_path, use_cuda=True, sample_rate=16000):
    """
    Generates audio from text using a TTS model.
    Input:
        tts_model: str, name of the TTS model
        data: str, input text
    Output:
        audio: np.array, generated audio
    """
    logger.info("Generating audio from token {}" .format(data))

    if config['mode'] == 'taipei1':
        model, vocoder = tts_model
        wav = tts_pipeline_taipei1(config, model, vocoder, model.h, "cuda" if use_cuda else "cpu", data, output_path)
        # resample to sample_rate
        wav = librosa.resample(wav, orig_sr=model.h.sampling_rate, target_sr=sample_rate)
        return wav
    
    if config['mode'] == 'cascade':
        wav = tts_model.text_to_wav(data, voice_name="alloy")
        sf.write(output_path, wav, sample_rate)
        return wav

    def dump_result(data, pred_wav):
        sf.write(
            output_path,
            pred_wav.detach().cpu().numpy(),
            sample_rate,
        )

    def code2wav(vocoder, codes, speaker_id, use_cuda=True):
        if isinstance(codes, str):
            codes = [int(c) for c in codes.strip(' ').split()]
        if isinstance(codes, list):
            codes = [int(c) for c in codes]

        if len(codes) > 0:
            inp = dict()
            logger.debug(f"codes: {codes}")
            inp["code"] = torch.LongTensor(codes).view(1, -1)
            inp["spkr"] = torch.LongTensor([speaker_id]).view(1, 1) 
            if use_cuda:
                inp = {k: v.cuda() for k, v in inp.items()}
            wav = vocoder(**inp).squeeze()
        return wav
    
    wav = code2wav(tts_model, data, 1, use_cuda=use_cuda)
    dump_result(data, wav)

    return wav.detach().cpu().numpy()

def load_tts_taipei1(config, device):
    with open(config['tts']['cfg'], 'r') as f:
        data = f.read()
    cfg = AttrDict(json.loads(data))
    model = Model(cfg).to(device)
    vocoder = Vocoder().to(device)

    state_dict = load_checkpoint(config['tts']['model_name'], device)
    model.load_state_dict(state_dict['model'])

    model.eval().remove_weight_norm()
    model.h = cfg

    return (model, vocoder)


# For Taipei1 model
def tts_pipeline_taipei1(config, model, vocoder, h, device, data, output_path):
    train_n_sch = torch.linspace(float(h.beta_0), float(h.beta_T), int(h.T)).to(device)
    dh = compute_hyperparams_given_schedule(train_n_sch)
    eval_n_sch = get_eval_noise_schedule(h.N, dh, device)

    # TODO: read units here
    # TODO: You have to duplicate the tokens!!
    # km = "310 310 112 112 237 237 411 411 197 197 390 390 121 121 171 171 197 197 492 492 492 492 20 20 20 20 119 119 428 428 189 189 157 157 15 15 153 153 353 353 378 378 116 116 374 374 88 88 498 498 204 204 310 310 157 157 72 72 498 498 189 189 73 73 411 411 134 134 498 498 316 316 498 498 299 299 299 299 498 498 498 498 498 498 498 498 243 243 268 268 362 362 335 335 164 164 21 21 498 498 242 242 493 493 223 223 423 423 498 498 104 104 419 419 193 193 281 281 498 498 92 92 498 498 250 250 239 239 498 498 498 498 498 498 266 266 169 169 243 243 293 293 498 498 168 168 49 49 498 498 293 293 498 498 399 399 303 303 377 377 408 408 424 424 294 294 297 297 341 341 436 436 217 217 390 390 114 114 395 395 450 450 164 164 21 21 192 192 363 363 493 493 201 201 210 210 153 153 3 3 164 164 498 498 362 362 457 457 498 498 8 8 424 424 315 315 391 391 498 498 498 498 112 112 351 351 21 21 498 498 405 405 498 498 334 334 292 292 498 498 116 116 499 499 204 204 53 53 11 11 311 311 498 498 395 395 368 368 498 498"
    # km = [int(k) for k in km.split()]
    km = [int(k) for k in data]
    km = torch.LongTensor(km)

    audio, sampling_rate = sf.read(config['tts']['ref_audio'])
    audio = audio / MAX_WAV_VALUE
    audio = normalize(audio) * 0.95
    if sampling_rate != h.sampling_rate:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=h.sampling_rate)

    audio = torch.FloatTensor(audio)
    align = len(km) * h.sampling_rate // h.km_rate
    if len(audio) > align:
        audio = audio[:align]
    elif len(audio) < align:
        audio = torch.nn.functional.pad(audio, (0, align - len(audio)), 'constant')
    assert len(audio) == align
    km = km.unsqueeze(0)
    audio = audio.unsqueeze(0)

    # print(audio.shape)
    # print(km.shape)
    with torch.no_grad():
        with torch.inference_mode():
            start = time.time()
            km = km.to(device)
            audio = audio.to(device)
            # from model.util import sampling_given_noise_schedule
            out = sampling_given_noise_schedule_ddim(h, eval_n_sch, model, km, audio)
            # print(out)
            middle = time.time()
            out = vocoder(out)
            end = time.time()

            sf.write(output_path, out[0].cpu().numpy().squeeze(), h.sampling_rate)

            print("sampling  : ", round(middle - start, 2))
            print("total     : ", round(end - start, 2))
    
    return out[0].cpu().numpy().squeeze()

def set_seed(seed: int):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def track2_audio_conversation(
        configA, configB, 
        asr_modelA, asr_modelB,
        slm_modelA, slm_modelB,
        tts_modelA, tts_modelB,
        promptA, promptB, 
        output_dir, output_prefix, 
        device,
        latency=1, turn=10):
    """
    Simulates a conversation between two models, generates audio files, and records the conversation in a JSON file.

    Input: 
        configA: dict, contains asr_model_name, slm_model_name, tts_model_name
        configB: dict, contains asr_model_name, slm_model_name, tts_model_name
        promptA: str, system prompt for Model A
        promptB: str, system prompt for Model B
        output_dir: str, directory to save audio files
        output_prefix: str, prefix for audio files
        latency: float, latency in seconds between the two models
        turn: int, number of turns in the conversation
    Output: Two audio files, one for each model's audio conversation, and a JSON file recording the conversation.
    """
    
    # Initialize conversation histories
    conversation_historyA = [{'role': 'system', 'content': promptA}]
    conversation_historyB = [{'role': 'system', 'content': promptB}]
    
    # For recording the conversation
    conversation_log = []

    # Initialize audio arrays
    audioA = []
    audioB = []

    # Set sample rate
    sample_rate = 16000

    logger.info("Starting conversation...")
    # Conversation loop
    for t in range(turn):
        # Model A's turn
        generated_text, tokens, text = slm_pipeline(configA, slm_modelA, conversation_historyA, 'A', device)

        # conversation_historyA.append({'role': 'assistant', 'content': generated_text})
        conversation_historyA.append({'role': 'Machine', 'content': generated_text})

        # Record Model A's response
        conversation_log.append({'speaker': 'Model A', 'text': text})

        # Generate audio from text using TTS model A
        wav = tts_pipeline(configA, tts_modelA, tokens, os.path.join(output_dir, f"{output_prefix}_A_{t}.wav"), use_cuda=device=='cuda', sample_rate=sample_rate)
        wav = wav

        # Append audio to Model A's output and silence to Model B's output
        audioA.append(wav)
        audioB.append(np.zeros_like(wav))

        # Add latency
        if latency > 0:
            latency_samples = int(latency * sample_rate)
            silence_latency = np.zeros(latency_samples)
            audioA.append(silence_latency)
            audioB.append(silence_latency)

        # Transcribe Model A's audio using ASR model B
        transcription, tokens, text = asr_unit_pipeline(configB, asr_modelB, os.path.join(output_dir, f"{output_prefix}_A_{t}.wav"))
        conversation_historyB.append({'role': 'user', 'content': transcription})

        # Record Model B's received input
        conversation_log.append({'speaker': 'Model B (heard)', 'text': text})

        # Model B's turn
        generated_text, tokens, text = slm_pipeline(configB, slm_modelB, conversation_historyB, 'B', device)
        conversation_historyB.append({'role': 'Machine', 'content': generated_text})

        # Record Model B's response
        conversation_log.append({'speaker': 'Model B', 'text': text})

        # Generate audio from text using TTS model B
        wav = tts_pipeline(configB, tts_modelB, tokens, os.path.join(output_dir, f"{output_prefix}_B_{t}.wav"), use_cuda=device=='cuda', sample_rate=sample_rate)
        wav = wav

        # Append silence to Model A's output and audio to Model B's output
        audioA.append(np.zeros_like(wav))
        audioB.append(wav)

        # Add latency
        if latency > 0:
            latency_samples = int(latency * sample_rate)
            silence_latency = np.zeros(latency_samples)
            audioA.append(silence_latency)
            audioB.append(silence_latency)

        # Transcribe Model B's audio using ASR model A
        transcription, tokens, text = asr_unit_pipeline(configA, asr_modelA, os.path.join(output_dir, f"{output_prefix}_B_{t}.wav"))
        conversation_historyA.append({'role': 'user', 'content': transcription})

        # Record Model A's received input
        conversation_log.append({'speaker': 'Model A (heard)', 'text': text})

    # Save the audio files
    audioA = np.concatenate(audioA)
    audioB = np.concatenate(audioB)
    os.makedirs(output_dir, exist_ok=True)
    output_pathA = os.path.join(output_dir, f"{output_prefix}_A.wav")
    output_pathB = os.path.join(output_dir, f"{output_prefix}_B.wav")
    sf.write(output_pathA, audioA, sample_rate)
    sf.write(output_pathB, audioB, sample_rate)

    # Save the conversation log to JSON
    conversation_file = os.path.join(output_dir, f"{output_prefix}_conversation.json")
    logger.debug(f"Conversation log: {conversation_log}")
    with open(conversation_file, 'w', encoding='utf-8') as f:
        json.dump(conversation_log, f, ensure_ascii=False, indent=4)

def load_asr(config, device):
    whisper = WhisperModel(config['asr']['model_name'], device=device, compute_type="float16")
    return whisper

def load_slm(config, device):
    if config["mode"] == "cascade":
        return None, None
    slm_model = AutoModelForCausalLM.from_pretrained(
        config['slm']['model_name'],
        token=config['slm']['hf_token'],
        cache_dir=config['slm']['cache_dir'],
        torch_dtype=torch.float16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        config['slm']['model_name'],
        cache_dir=config['slm']['cache_dir'],
        token=config['slm']['hf_token'],
    )
    return slm_model, tokenizer

def load_tts(config):
    if config['mode'] == 'taipei1':
        return load_tts_taipei1(config, device='cuda')
    if config['mode'] == 'cascade':
        api = OpenAITTSAPI(sr=16000)
        return api
    def load_vocoder(cfg, filepath):
        with open(cfg, 'r') as f:
            vocoder_cfg = AttrDict(json.load(f))
        vocoder = CodeGenerator(vocoder_cfg)
        assert os.path.isfile(filepath)
        print("Loading '{}'".format(filepath))
        state_dict_g = torch.load(filepath, map_location='cpu')
        print("Complete.")
        vocoder.load_state_dict(state_dict_g['generator'])
        if torch.cuda.is_available():
            tts_model = vocoder.cuda()
        return tts_model

    tts_model = load_vocoder(config['tts']['vocoder_cfg'], config['tts']['model_name'])
    return tts_model

def track2_inference(configA, configB, prompt_path, output_dir, output_prefix, latency=1, turn=5):
    """
    Simulates a conversation between two models, generates audio files, and records the conversation in a JSON file.

    Input: 
        configA: dict, contains asr_model_name, slm_model_name, tts_model_name
        configB: dict, contains asr_model_name, slm_model_name, tts_model_name
        prompt_path: str, path to the prompt dataset file
        output_dir: str, directory to save audio files
        output_prefix: str, prefix for audio files
        latency: float, latency in seconds between the two models
        turn: int, number of turns in the conversation
    """

    # Set your OpenAI API key
    if configA['use_openai'] or configB['use_openai']:
        os.environ["OPENAI_API_KEY"] = configA['openai_api_key'] if configA['use_openai'] else configB['openai_api_key']
        assert os.environ.get("OPENAI_API_KEY") is not None, "Please set your OpenAI API key as an environment variable."

    # Load the prompt dataset
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_dataset = json.load(f)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    set_seed(seed=1126)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)


    # Load ASR models
    if configA['asr']['model_name'] == configB['asr']['model_name']:
        asr_model = load_asr(configA, device)
        asr_modelA = asr_modelB = asr_model
    else:
        asr_modelA = load_asr(configA, device)
        asr_modelB = load_asr(configB, device)
    
    # Load SLM models
    if configA['slm']['model_name'] == configB['slm']['model_name']:
        slm_model, tokenizer = load_slm(configA, device)
        slm_modelA = slm_modelB = (slm_model, tokenizer)
    else:
        slm_modelA, tokenizerA = load_slm(configA, device)
        slm_modelB, tokenizerB = load_slm(configB, device)
        slm_modelA = (slm_modelA, tokenizerA)
        slm_modelB = (slm_modelB, tokenizerB)
    
    
    # Load TTS models
    if configA['tts']['model_name'] == configB['tts']['model_name']:
        tts_model = load_tts(configA)
        tts_modelA = tts_modelB = tts_model
    else:
        tts_modelA = load_tts(configA)
        tts_modelB = load_tts(configB)

    # Loop through the prompts
    for idx, prompts in enumerate(prompt_dataset, start=1):
        promptA = prompts['system_prompt_A'] + " Modality: {{User: speech, Machine: speech}}. Speech Style: Audio Book."
        promptB = prompts['system_prompt_B'] + " Modality: {{User: speech, Machine: speech}}. Speech Style: Audio Book."
        output_prefix_idx = f"{output_prefix}_{idx:04d}"

        logger.info(f"Starting conversation {output_prefix_idx}...")
        
        track2_audio_conversation(
            configA=configA,
            configB=configB,
            asr_modelA=asr_modelA,
            asr_modelB=asr_modelB,
            slm_modelA=slm_modelA,
            slm_modelB=slm_modelB,
            tts_modelA=tts_modelA,
            tts_modelB=tts_modelB,
            promptA=promptA,
            promptB=promptB,
            output_dir=output_dir,
            output_prefix=output_prefix_idx,
            device=device,
            latency=latency,
            turn=turn
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configA', type=str, default='offline_record_config/config_cascade.yaml', help='Path to the config file for Model A')
    parser.add_argument('--configB', type=str, default='offline_record_config/config_taipei1.yaml', help='Path to the config file for Model B')
    parser.add_argument('--prompt_path', type=str, default='track2_chinese_prompt.json', help='Path to the prompt dataset file')
    parser.add_argument('--output_dir', type=str, default='./conversation_outputs', help='Path to the output directory')
    args = parser.parse_args()
    
    # Load the YAML file
    with open(args.configA, 'r') as f:
        configA = yaml.safe_load(f)
    with open(args.configB, 'r') as f:
        configB = yaml.safe_load(f)

    logger.info(f"Config A: {configA}")
    logger.info(f"Config B: {configB}")

    track2_inference(
        configA=configA,
        configB=configB,
        prompt_path=args.prompt_path,
        output_dir=args.output_dir,
        output_prefix=args.prompt_path.split('.')[0].split('_')[1],
        latency=1,
        turn=3
    )


    # promptA = "You are a helpful assistant." 
    # # 你是一個能和人類溝通與交流的聊天機器人。今天你正在和使用者進行一段有趣的對談
    # promptB = "You are an inquisitive user."

    # promptA += " Modality: {{User: speech, Machine: speech}}. Speech Style: Audio Book."
    # promptB += " Modality: {{User: speech, Machine: speech}}. Speech Style: Audio Book."

    


    