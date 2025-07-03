import json
from collections import defaultdict
from pathlib import Path

CUB = Path("~/NSERC/data/CUB_200_2011").expanduser()

img_names = {int(i) : name for i, name in 
             (l.strip().split(maxsplit=1) for l in open(CUB / "CUB_200_2011/images.txt"))}

attr_names = {int(i) : name for i, name in
              (l.strip().split(maxsplit=1) for l in open(CUB / "attributes.txt"))}

certainty = {int(i) : degree for i, degree in
             (l.strip().split(maxsplit=1) for l in open(CUB / "CUB_200_2011/attributes/certainties.txt"))}

per_image = defaultdict(lambda: defaultdict(list))

with open(CUB / "CUB_200_2011/attributes/image_attribute_labels.txt") as f :
    for line in f :
        parts = line.split()
        
        img_id, attr_id, is_present, cert_id = parts[:4]

        img_key = img_names[int(img_id)]
        attr_name = attr_names[int(attr_id)]
        rec = {
            "is_present" : bool(int(is_present)),
            "certainty" : certainty[int(cert_id)],
        }

        per_image[img_key][attr_name].append(rec)
    
out = Path("~/NSERC/attributes.json").expanduser()

with out.open("w") as f :
    json.dump(per_image, f, indent=2)
