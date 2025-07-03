import json
from pathlib import Path
from collections import defaultdict

root = Path("~/NSERC/data/generated_captions/CUB_captions").expanduser()

captions = defaultdict(list)          # {"011.Clay_.../Clay_..._0001_...": [sentences,…]}

for class_dir in root.iterdir():
    if not class_dir.is_dir():
        continue                      # skip non-folders

    folder_name = class_dir.name      # e.g. "011.Clay_Colored_Sparrow"

    # walk every *.txt file under this class directory
    for txt_path in class_dir.rglob("*.txt"):
        file_stem  = txt_path.stem    # "Clay_Colored_Sparrow_0001_110635"
        full_key   = f"{folder_name}/{file_stem}"

        with txt_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line and line not in captions[full_key]:
                    captions[full_key].append(line)

# dump as a flat JSON map
save_path = root / "CUB_captions.json"
with save_path.open("w", encoding="utf-8") as f:
    json.dump(captions, f, indent=2)

print("Wrote", save_path)
# import json, sys, re, pathlib
# path = pathlib.Path("CUB_captions.json")
# data = json.load(path.open())

# word_re = re.compile(r"\w+")
# total_words = total_sents = 0

# for bird in data.values() :
#     for caption_list in bird["captions"].values():
#         for sent in caption_list :
#             total_words += len(word_re.findall(sent))
#             total_sents += 1

# print(f"Average words per sentence {total_words / total_sents:.2f}")