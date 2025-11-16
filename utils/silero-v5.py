#!/usr/bin/env python3

import torch
import sounddevice as sd
import abbrev_fix
import re
import sys
from textwrap import wrap

# Get the first command-line argument, or default to
# an empty string if none is provided
sample_text_file = sys.argv[1] if len(sys.argv) > 1 else ""

LANG = 'en'

if LANG == 'ru':
    language = 'ru'
    model_id = 'v5_ru'
    speaker = 'baya'
else:
    # language = 'multi'
    # model_id = 'multi_v2'
    language = 'en'
    model_id = 'v3_en'
    speaker = 'en_91'

sample_rate = 48000
# aidar, baya, kseniya, xenia, eugene
put_accent = True
put_yo = True
put_stress_homo = True
put_yo_homo = True

device = torch.device('cuda')  # type: ignore

model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language=language,
                                     put_accent=put_accent,
                                     put_yo=put_yo,
                                     put_stress_homo=put_stress_homo,
                                     put_yo_homo=put_yo_homo,
                                     speaker=model_id)  # type: ignore
model.to(device)  # gpu or cpu

# Check if the sample_text_file is provided and exists
if sample_text_file:
    try:
        with open(sample_text_file, 'r') as file:
            example_text = file.read()
    except FileNotFoundError:
        print(f"File '{sample_text_file}' does not exist. Using default text.")
else:
    example_text = """
NVIDIA announces collaborations and investments in AI supercomputing,
quantum computing, and 6G development with partners such as
Oracle, Nokia, Palantir, and Uber.
"""


# Split the text into sentence blocks of less than 5000 characters each
def split_into_sentence_blocks(text, max_length=1000):
    sentences = re.split(r'(?<=[.!?])\s+|\n\s*\n', text, flags=re.MULTILINE)
    sentence_blocks = []
    current_block = ""

    for sentence in sentences:
        if len(sentence) > max_length:
            for word in sentence.split():
                if len(current_block) + len(word) > max_length:
                    sentence_blocks.append(current_block.strip())
                    current_block = ""

        if len(current_block) + len(sentence) > max_length:
            sentence_blocks.append(current_block.strip())
            current_block = ""

        current_block += sentence + " "

    if current_block:
        sentence_blocks.append(current_block.strip())

    return sentence_blocks


# Additional processing on example_text
example_text = re.sub(r'\bNVIDIA\b', 'En Vee Dee Ahh', example_text)
example_text = abbrev_fix.replace_capital_letters(example_text)

sentence_blocks = split_into_sentence_blocks(example_text, 700)


print(f"Setenses block len={len(sentence_blocks)}")
wait_next = False
for block in sentence_blocks:
    print(f"{len(block)}\n\n===============\n")
    print(block)

    if wait_next:
        sd.wait()
    audio = model.apply_tts(text=block,
                            speaker=speaker,
                            sample_rate=sample_rate)
    sd.play(audio, sample_rate)
    wait_next = True
sd.wait()
