import evaluate 
from torchtext.data.metrics import bleu_score
import json
import sys

def main() :
    ref_path = sys.argv[1]
    pred_path = sys.argv[2]

    print("Loading reference from:", ref_path)
    print("Loading prediction from:", pred_path)

    with open(ref_path, "r") as f:
        ref_file = json.load(f)
    with open(pred_path, "r") as f:
        pred_file = json.load(f)

    bleu = evaluate.load("bleu")

    predictions = []
    references = []

    for id in ref_file:
        predictions.append(pred_file[id])     # string
        references.append(ref_file[id])       # list of strings

    results = bleu.compute(predictions=predictions, references=references, max_order=4)
    print("BLEU-4 Score:", results['bleu'])

if __name__ == "__main__" :
    main()