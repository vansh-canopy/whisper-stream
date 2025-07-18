import torch
import torchaudio
import wandb
from transformers import Trainer, TrainingArguments, WhisperProcessor # type: ignore[attr-defined]
from utils import prepare_data, load_model
import inspect

#first create whisper model
processor = WhisperProcessor.from_pretrained("openai/whisper-base")

from transformers import WhisperModel, WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder

class CausalWhisperEncoder(WhisperEncoder):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)

    def _create_causal_mask(self, seq_length, device, dtype):
        mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.unsqueeze(0).unsqueeze(0).to(dtype)
    
    def forward(self, input_features, causal_mask, **kwargs):
        # Run the standard preprocessing
        inputs_embeds = torch.nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = torch.nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        
        # Positional encoding
        all_positions = torch.arange(self.embed_positions.num_embeddings, device=inputs_embeds.device)
        hidden_states = inputs_embeds + self.embed_positions(all_positions)
        hidden_states = torch.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        
        # Process through layers with mask
        encoder_states = () if kwargs.get('output_hidden_states', False) else None
        all_attentions = () if kwargs.get('output_attentions', False) else None
        
        for idx, encoder_layer in enumerate(self.layers):
            if kwargs.get('output_hidden_states', False):
                encoder_states = encoder_states + (hidden_states,)
            
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                layer_head_mask=kwargs.get('head_mask', [None]*len(self.layers))[idx],
                output_attentions=kwargs.get('output_attentions', False),
            )
            
            hidden_states = layer_outputs[0]
            
            if kwargs.get('output_attentions', False):
                all_attentions = all_attentions + (layer_outputs[1],)
        
        hidden_states = self.layer_norm(hidden_states)
        
        if kwargs.get('output_hidden_states', False):
            encoder_states = encoder_states + (hidden_states,)
        
        from transformers.modeling_outputs import BaseModelOutput
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions
        )

class CausalWhisperModel(WhisperModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        # Replace the encoder with our causal version
        self.encoder = CausalWhisperEncoder(config)

# Usage
config = WhisperConfig.from_pretrained("openai/whisper-base")
model = CausalWhisperModel.from_pretrained("openai/whisper-base", config=config)

print("Causal Whisper model created with causal mask support.")


import torch
import numpy as np

# Create random input that matches Whisper's expected format
def get_random_encoded_latents(model, random_input, causal_mask=None):

    with torch.no_grad():
        encoder_outputs = model.encoder(
            random_input,
            return_dict=True,
            causal_mask=causal_mask
        )
    
    latents = encoder_outputs.last_hidden_state
    
    return latents

# Usage
model = CausalWhisperModel.from_pretrained("openai/whisper-base", config=config)
model.eval()

n_mels = 80
n_frames = 3000  # This is the expected length for 30s audio

# Create random input features
shared_frames = torch.randn(1, n_mels, 1000)
sample_1 = torch.cat([shared_frames, torch.randn(1, n_mels, n_frames - 1000)], dim=2)
sample_2 = torch.cat([shared_frames, torch.randn(1, n_mels, n_frames - 1000)], dim=2)



mask = torch.triu(torch.ones(1500, 1500), diagonal=1)
mask = mask.masked_fill(mask == 1, float('-inf'))
latents_1 = get_random_encoded_latents(model, sample_1, causal_mask=mask)
latents_2 = get_random_encoded_latents(model, sample_2, causal_mask=mask)

print(latents_1.shape, latents_2.shape)

latents_delta = latents_1 - latents_2

print("Latent delta shape:", latents_delta.shape)

print(latents_delta[0][498])
