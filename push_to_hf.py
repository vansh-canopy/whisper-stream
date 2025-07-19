from causal_wrapper import CausalWhisperForConditionalGeneration
from transformers import WhisperConfig

# Make sure to check the look ahead is the same as the one used to train the model

MODEL_ID = "openai/whisper-base"
CHECKPOINT_To_SAVE = "whisper_look_ahead_run/checkpoint-100000"
LOOK_AHEAD = 3

conf = WhisperConfig.from_pretrained(MODEL_ID)
causal_model = CausalWhisperForConditionalGeneration(conf).from_pretrained(CHECKPOINT_To_SAVE)

# This should save the correct causal mask as well, so when you load the model somehwere else
# you don't have to create it again. Not fully sure...
causal_model.model.encoder.causal_mask = causal_model.model.encoder._create_lookahead_mask(1500,  LOOK_AHEAD)


causal_model.push_to_hub("vanshjjw/whisper-stream-lookahead-3")
