import os
import sys

# Run as a script (deepspeed llava/train/train_mem.py): make the repo root importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
    #train(attn_implementation=None)
