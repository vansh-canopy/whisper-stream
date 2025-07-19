import torch
import torchaudio
from causal_wrapper import CausalWhisperForConditionalGeneration
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperConfig
from utils import prepare_data

MODEL_ID = "openai/whisper-base"
CHECKPOINT_PATH = "whisper_look_ahead_1/checkpoint-100000"
DEVICE = "cuda:5"
LOOK_AHEAD = 3


def main():
    processor = WhisperProcessor.from_pretrained(MODEL_ID)

    base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)
    base_model.eval()

    conf = WhisperConfig.from_pretrained(MODEL_ID)
    causal_model = CausalWhisperForConditionalGeneration(conf).from_pretrained(CHECKPOINT_PATH)
    causal_model.model.encoder.causal_mask = causal_model.model.encoder._create_lookahead_mask(1500,  LOOK_AHEAD)
    causal_model.to(DEVICE).eval()
    print("Loaded models")

    ds = prepare_data(max_shards=1)
    
    CAUSAL_LOSSES = []
    BASE_LOSSES = []
    NUM_SAMPLES_TO_COMPARE = 100
    PRINT_TEXT = True
    
    for i in range(NUM_SAMPLES_TO_COMPARE):
        
        if i % 10 == 0:
            print(f"Comparing sample {i} of {NUM_SAMPLES_TO_COMPARE}")
        
        sample = ds[i]

        audio_array = sample["mp3"]["array"]
        sr = sample["mp3"]["sampling_rate"]

        audio = torch.from_numpy(audio_array).float()
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            audio = resampler(audio)

        # Prepare inputs and labels
        text = sample["json"]["text"]
        labels = torch.tensor(processor.tokenizer(text, add_special_tokens=False).input_ids).unsqueeze(0).to(DEVICE)

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        input_feats = inputs.input_features.to(DEVICE)

        with torch.no_grad():
            causal_out = causal_model(input_feats, labels=labels)
            base_out = base_model(input_feats, labels=labels)

            causal_ids = causal_model.generate(input_feats)
            base_ids = base_model.generate(input_feats)

        if PRINT_TEXT:
            causal_answer = processor.batch_decode(causal_ids, skip_special_tokens=True)[0]
            base_answer = processor.batch_decode(base_ids, skip_special_tokens=True)[0]
            print(f"--------------------------------")
            print("GROUND TRUTH:", text)
            print("CAUSAL MODEL:", causal_answer)
            print("BASE MODEL:", base_answer)
            print(f"--------------------------------")
        
        CAUSAL_LOSSES.append(causal_out.loss.item())
        BASE_LOSSES.append(base_out.loss.item())

    print(f"Causal loss: {sum(CAUSAL_LOSSES) / len(CAUSAL_LOSSES):.4f}")
    print(f"Base loss: {sum(BASE_LOSSES) / len(BASE_LOSSES):.4f}")
            
        

if __name__ == "__main__":
    main() 