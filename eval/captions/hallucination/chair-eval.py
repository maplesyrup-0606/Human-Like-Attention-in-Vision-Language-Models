import json, argparse, os, csv
from pathlib import Path

def save_chair_results(results_dir, save_dir):
    results_dir = Path(results_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    output_csv = save_dir / "chair_overall_results.csv"

    os.makedirs(save_dir, exist_ok=True)

    rows = []

    for chair_results in results_dir.rglob("chair_*.json") :
        chair_results = Path(chair_results).expanduser()
        chair_json = json.load(open(chair_results, "r"))
        method_name = chair_results.stem.strip('chair_')
        method_results = chair_json["overall_metrics"]

        row = {
            "Method" : method_name,
            **method_results
        }
        rows.append(row)
    
    fieldnames = ["Method"] + sorted({k for row in rows for k in row if k != "Method"})

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Saved!")
    return 

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()

    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--save-dir", required=True)

    args = parser.parse_args()

    save_chair_results(args.results_dir, args.save_dir)