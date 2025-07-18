import torch
import numpy as np
import os
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset               # type: ignore

def load_model(causal=True):
    model_id = "openai/whisper-base"
    wrapper = WhisperForConditionalGeneration.from_pretrained(model_id)
    
    if causal:
        for layer in wrapper.model.encoder.layers:        
            layer.self_attn.is_causal = True          # type: ignore[attr-defined]
          
    processor = WhisperProcessor.from_pretrained(model_id)
    return wrapper, processor


def run_inference(audio, model, processor):
    # Process audio
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    
    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(inputs.input_features)
    
    # Decode transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    return transcription


def quick_check():
    wrapper, processor = load_model()
    
    # Random audio (5 seconds at 16kHz)
    audio = np.random.randn(1600 * 9)
    text = "This is a placeholder sentence."
    
    answer = run_inference(audio, wrapper, processor)
       
    # Compare with ground truth
    print(f"Ground truth: {text}")
    print(f"Transcription: {answer}")
    
    
def prepare_data(load_from="/mnt/disks/emilia/emilia_dataset/Emilia/EN", max_shards=2000, num_proc=200):
    tar_paths = sorted([filename for filename in os.listdir(load_from) if filename.endswith(".tar")])
    language = "en"

    selected_tar_paths = tar_paths[:max_shards]
    data_files = {language: selected_tar_paths}

    ds = load_dataset(  
        load_from,
        data_files=data_files,
        split=language,
        num_proc=num_proc,
        cache_dir="/mnt/disks/emilia/emilia_cache/"
    )
    
    return ds.remove_columns([c for c in ds.column_names if c not in ["mp3", "json"]])  # type: ignore[attr-defined]
