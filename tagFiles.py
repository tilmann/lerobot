from huggingface_hub import HfApi
api = HfApi()
for slot in range(1, 8):
    api.create_tag(f"tilmannb/4inarow_slot{slot}", tag="v3.0", repo_type="dataset")