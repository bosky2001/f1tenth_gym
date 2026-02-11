import torch
from train_perception_map import PerceptionMap

model_path = 'perception_model.pth'

# kaiming he init 

n_input = 1083
n_hidden = 2048
n_output = 3
use_pos_encoding = False  # Multi-frequency trigonometric positional encoding
n_frequencies = 4  # Number of frequency bands (1, 2, 4, 8)
dropout = 0.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = PerceptionMap(n_input, n_hidden, n_output, use_pos_encoding=use_pos_encoding,
                          n_frequencies=n_frequencies, dropout=dropout)

state_dict = torch.load(model_path, map_location=device)
# Handle torch.compile() prefix if present
if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)

all_params_list = list(model.parameters())
print(all_params_list)