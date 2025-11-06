from huggingface_hub import hf_hub_download
# LJSpeech
hf_hub_download("yl4579/StyleTTS2-LJSpeech","Models/LJSpeech/config.yml",local_dir=".")
hf_hub_download("yl4579/StyleTTS2-LJSpeech","Models/LJSpeech/epoch_2nd_00100.pth",local_dir=".")
# LibriTTS
hf_hub_download("yl4579/StyleTTS2-LibriTTS","Models/LibriTTS/config.yml",local_dir=".")
hf_hub_download("yl4579/StyleTTS2-LibriTTS","Models/LibriTTS/epochs_2nd_00020.pth",local_dir=".")
hf_hub_download("yl4579/StyleTTS2-LibriTTS","reference_audio.zip",local_dir=".")