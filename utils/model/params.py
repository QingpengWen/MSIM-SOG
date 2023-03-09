from utils.model.module import ModelManager
from utils.data_loader.loader import DatasetManager
from utils.config import *

args = parser.parse_args()

dataset = DatasetManager(args)
dataset.quick_build()
# dataset.show_summary()
model_fn = ModelManager
# Instantiate a network model object.
model = model_fn(
    args, len(dataset.char_alphabet),
    len(dataset.word_alphabet),
    len(dataset.slot_alphabet),
    len(dataset.intent_alphabet)
)


num = 0
for i in model.parameters():
    if i.requires_grad == True:
        num += i.numel()
print(num)