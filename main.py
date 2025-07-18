import torch
import torchaudio
import wandb
from transformers import Trainer, TrainingArguments # type: ignore[attr-defined]
from utils import prepare_data, load_model


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


def train():
    wandb.init(project="whisper-causal-round-3")

    print("Loading data...")
    ds = prepare_data(max_shards=1000)   # type: ignore[attr-defined]

    print("Loading model...")
    model, processor = load_model(causal=True)

    data_collator = WhisperDataCollator(processor)

    training_args = TrainingArguments(
        output_dir="./whisper_causal_3",
        per_device_train_batch_size=8,
        num_train_epochs=1,
        logging_steps=100,
        save_steps=1000,
        eval_steps=1000,
        warmup_steps=500,
        save_total_limit=10,
        learning_rate=1e-4,
        remove_unused_columns=False,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,                   # type: ignore[attr-defined]
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()


if __name__ == "__main__":
    train()