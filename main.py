from transformers import WhisperForConditionalGeneration
from huggingface_hub import HfApi

def main():
    # Load and push model
    model = WhisperForConditionalGeneration.from_pretrained("./whisper_causal_3/checkpoint-103000", local_files_only=True)
    model.push_to_hub(repo_id="vanshjjw/whisper-stream")  # type: ignore
    
    # Upload complete checkpoint folder (includes optimizer, schedule   r, etc.)
    api = HfApi()
    api.upload_folder(
        folder_path="./whisper_causal_3/checkpoint-103000",
        repo_id="vanshjjw/whisper-stream",
        path_in_repo="checkpoints/"
    )
    
    print("Complete checkpoint pushed to HF!")

if __name__ == "__main__":
    main()
