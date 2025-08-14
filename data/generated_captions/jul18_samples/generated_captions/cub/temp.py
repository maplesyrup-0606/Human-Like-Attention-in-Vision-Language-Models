import json
from pathlib import Path

baseline = Path("baseline.json").expanduser()
baseline = json.load(baseline.open())

total_word_count = 0 
total_sentences = 0

for id_, captions in baseline.items() :
    caption = captions[0]
    total_sentences += 1
    caption = caption.split(" ")
    total_word_count += len(caption)

print(f"Average word count {total_word_count / total_sentences}")