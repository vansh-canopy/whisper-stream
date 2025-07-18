import torch
import torchaudio
import wandb
from transformers import Trainer, TrainingArguments, WhisperProcessor # type: ignore[attr-defined]
from utils import prepare_data, load_model
from causal_wrapper import load_causal_whisper


class WhisperDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = []
        labels = []

        for feature in features:
            audio = torch.from_numpy(feature["mp3"]["array"]).float()
            sr = feature["mp3"]["sampling_rate"]
            text = feature["json"]["text"]

            if sr != 16000:
                audio = torchaudio.transforms.Resample(sr, 16000)(audio)

            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
            input_features.append(inputs.input_features[0])

            labels.append(self.processor.tokenizer(text, return_tensors="pt").input_ids[0])

        batch = {
            "input_features": torch.stack(input_features),
            "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        }
        return batch


def run_lookahead():
    LOOK_AHEAD = 3  
    NUM_LATENTS = 1500
    MODEL_ID = "openai/whisper-base"
    
    print("Loading data...")
    ds = prepare_data(max_shards=1200)   # type: ignore[attr-defined]

    print("Loading model...")
    model = load_causal_whisper(MODEL_ID, for_conditional=True)
    model.model.encoder.causal_mask = model.model.encoder._create_lookahead_mask(NUM_LATENTS, LOOK_AHEAD)
    processor = WhisperProcessor.from_pretrained(MODEL_ID)

    data_collator = WhisperDataCollator(processor)
    
    learning_rate = 5e-5
    per_device_batch_size = 4
    num_epochs = 1
    
    wandb.init(
        project=f"whisper-lookahead-{LOOK_AHEAD}", 
        name=f"lr-{learning_rate}-bs-{per_device_batch_size}-epochs-{num_epochs}"
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.01)

    training_args = TrainingArguments(
        output_dir="./whisper_look_ahead_1",
        per_device_train_batch_size=per_device_batch_size,
        num_train_epochs=num_epochs,
        logging_steps=10,
        save_steps=1000,
        eval_steps=1000,
        warmup_steps=500,
        save_total_limit=20,
        learning_rate=1e-4,
        remove_unused_columns=False,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,    # type: ignore[attr-defined]
        data_collator=data_collator,
        optimizer=(optimizer, None)
    )

    print("Starting training...")
    trainer.train()



def run_causal():
    print("Loading data...")
    ds = prepare_data(max_shards=10)   # type: ignore[attr-defined]

    print("Loading model...")
    model, processor = load_model(causal=True)

    data_collator = WhisperDataCollator(processor)
    
    learning_rate = 1e-4
    per_device_batch_size = 4
    num_epochs = 1
    
    wandb.init(
        project=f"whisper-causal", 
        name=f"lr-{learning_rate}-bs-{per_device_batch_size}-epochs-{num_epochs}"
    )

    training_args = TrainingArguments(
        output_dir="./whisper_look_ahead_1",
        per_device_train_batch_size=per_device_batch_size,
        num_train_epochs=num_epochs,
        logging_steps=10,
        save_steps=1000,
        eval_steps=1000,
        warmup_steps=500,
        save_total_limit=20,
        learning_rate=1e-4,
        remove_unused_columns=False,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,    # type: ignore[attr-defined]
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()
    

if __name__ == "__main__":
    run_lookahead()