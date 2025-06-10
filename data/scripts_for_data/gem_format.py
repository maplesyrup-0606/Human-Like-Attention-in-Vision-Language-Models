import os
import json

def main() :
    ref_path = os.path.expanduser("~/NSERC/data/generated_captions/jun5_samples/sampled_captions.json")
    pred_path = os.path.expanduser("~/NSERC/data/generated_captions/may26_samples/gaussian_answered_captions_wordcap.json")

    with open(ref_path, "r") as f :
        ref_captions = json.load(f)
    
    with open(pred_path, "r") as f :
        pred_captions = json.load(f)
    

    ref_gem = {}
    pred_gem = {}

    pred_gem["values"] = []
    ref_gem["values"] = []

    for id in ref_captions :
        gt_captions = ref_captions[id]
        predictions = pred_captions[id]
        
        for i in range(len(predictions)) :
            ref_gem["values"].append({ "target" : gt_captions })
            pred_gem["values"].append(predictions[i])

    ref_gem["language"] = "en"
    pred_gem["language"] = "en"

    ref_save_path = os.path.expanduser("~/NSERC/data/generated_captions/jun5_samples/sampled_captions_gem.json")
    pred_save_path = os.path.expanduser("~/NSERC/data/generated_captions/may26_samples/gaussian_answered_captions_wordcap_gem.json")

    with open(ref_save_path, "w") as f :
        json.dump(ref_gem ,f, indent=2)
    
    with open(pred_save_path, "w") as f :
        json.dump(pred_gem, f, indent=2)

if __name__ == "__main__" :
    main()