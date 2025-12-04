import os
import yaml
import torch
from transformers import AlbertConfig, AlbertModel

class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        return outputs.last_hidden_state

def load_plbert(log_dir):
    # read config from the weights folder
    config_path = os.path.join(log_dir, "config.yml")
    plbert_config = yaml.safe_load(open(config_path, encoding="utf-8"))

    albert_base_configuration = AlbertConfig(**plbert_config['model_params'])
    bert = CustomAlbert(albert_base_configuration)

    # pick latest step_*.t7
    ckpts = [f for f in os.listdir(log_dir) if f.startswith("step_")]
    if not ckpts:
        raise FileNotFoundError(f"No PL-BERT checkpoints found in {log_dir}")
    iters = sorted(int(f.split('_')[-1].split('.')[0]) for f in ckpts
                   if os.path.isfile(os.path.join(log_dir, f)))
    step = iters[-1]

    checkpoint = torch.load(os.path.join(log_dir, f"step_{step}.t7"), map_location='cpu')
    state_dict = checkpoint['net']

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # drop leading "module."
        if name.startswith('encoder.'):
            name = name[8:]  # drop leading "encoder."
            new_state_dict[name] = v

    # ✅ safe for multilingual checkpoints (key may not exist)
    new_state_dict.pop("embeddings.position_ids", None)

    # ✅ tolerate harmless key diffs
    bert.load_state_dict(new_state_dict, strict=False)

    print(f"[PLBERT] loaded step_{step} from {os.path.abspath(log_dir)}")
    return bert
