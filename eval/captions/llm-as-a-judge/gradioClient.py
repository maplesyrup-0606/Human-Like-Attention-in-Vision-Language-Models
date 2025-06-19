import json
import sys
import os
from tqdm import tqdm
from judge_prompt import JUDGE_PROMPT
from gradio_client import Client

client = Client("http://127.0.0.1:7860")

gt_caption_file_path = os.path.expanduser(sys.argv[1])
gen_caption_file_path = os.path.expanduser(sys.argv[2])

file_name = os.path.basename(gen_caption_file_path)
base_name = os.path.splitext(file_name)[0]

rating_save_dir = os.path.expanduser(sys.argv[3])
os.makedirs(rating_save_dir, exist_ok=True)

with open(gt_caption_file_path, "r") as f :
    gt_captions = json.load(f)

with open(gen_caption_file_path, "r") as f :
    gen_captions = json.load(f)

ids = gt_captions.keys()
ratings = {}

batched_prompts = []
image_keys = []
for image_id in tqdm(ids, desc="Evaluating", file=sys.stdout) :
    cur_gen_captions = gen_captions[image_id]
    cur_gt_caption = gt_captions[image_id]

    responses = []
    for i, generated in enumerate(cur_gen_captions) :
        prompt = JUDGE_PROMPT.format(
            groundtruth=cur_gt_caption,
            generated=generated
        )
        
        response = client.predict(
            prompt,
            0.7,
            api_name=None
        )

        responses.append(response)

    ratings[image_id] = {
        'ratings' : responses
    }
    # ratings[image_id] = {
    #     'max_rating' : 0,
    #     'ratings' : []
    # }

    # for response in responses :
    #     feedback, rating = response.split("Total rating:")
    #     rating = float(rating)

    #     ratings[image_id]['max_rating'] = max(ratings[image_id]['max_rating'], rating)
    #     ratings[image_id]['ratings'].append({
    #         "feedback" : feedback,
    #         "rating" : rating
    #     })
    print(f"Response Saved for {image_id} ✅", flush=True)

with open(os.path.join(rating_save_dir, base_name + "_rating.json"), "w") as f :
    json.dump(f, ratings, indent=2)

print(f"Done Rating {gen_caption_file_path}",flush=True)