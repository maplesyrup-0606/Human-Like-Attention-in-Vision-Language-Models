JUDGE_PROMPT = """
You will be given a **generated caption** and a **set of ground truth captions** that describe a certain image.
You will also be given an example for each rating that you will need to assign as a reference.
Your task is to rate how well the generated caption matches or captures the overall meaning conveyed by the set of ground truth captions. This rating is called the **"Total rating"**, and it must be a number from **1 to 5**, where:

- **1** = Completely irrelevant
- **5** = Perfectly aligned and detailed

### Rating Scale:

**Score 1 – Completely Irrelevant and Vague**  
- Does not relate to any of the ground truth captions  
- May be nonsensical or describe unrelated content  
- Lacks meaningful or specific details

eg. **Generated** : "Red turbo diesel four-wheel drive vehicle with a sheep looking out of the window."
    **Groundtruths** : [
                    "A dog driving an SUV in an open grass covered field.",
                    "A dog sits in the front seat of a jeep. ",
                    "A dog is sitting inside a red car.",
                    "A red truck has a black dog in the drivers chair.",
                    "A dog sitting in the front seat of a truck."
                  ]

**Score 2 – Minimally Relevant, Low Detail**  
- Mentions a correct object or scene type, but content is mostly inaccurate  
- Omits most key elements or relationships  
- Very generic or vague

eg. **Generated** : "A busy urban intersection with a building on each corner."
    **Groundtruths** : [
                    "There is very little traffic at this city intersection.",
                    "An urban intersection with stoplights on a cloudy day.",
                    "a city street with multiple bildings and a street light",
                    "A four cross street of a downtown area.",
                    "A wide empty street corner filled with buildings."
                   ]

**Score 3 – Moderately Relevant and Informative**  
- Captures the general idea of at least one ground truth  
- Some objects, actions, or relationships are present, but lacks completeness  
- Understandable, but misses finer or supporting details

eg. **Generated** : "A black motorcycle parked on the edge of a street."
    **Groundtruths** : [
                    "A customized motorcycle with more in the background.",
                    "A black and white photograph of a motorcycle.",
                    "Black and white photo of a parked motorcycle",
                    "Several custom made motorcycles on display while riders chat.",
                    "A customized motorcycle with a large rear and skinny front tire"
                   ]
     
**Score 4 – Mostly Relevant and Detailed**  
- Describes most important elements reflected in multiple ground truth captions  
- Includes relevant attributes, actions, and relationships  
- May have minor omissions or inaccuracies

eg. **Generated** : "Two bronze statues of women sitting on a bench, one has a purse next to her."
    **Groundtruths** : [
                    "A sculpture of two women stting on a bench with their purses on the ground while people standing in a line behind them. ",
                    "A metal statue of two women sits on a bench in a city street.",
                    "A statue of two women with purses sitting on a bench. ",
                    "A statue of two people sitting on a bench.",
                    "A metal statue of two women sitting on a bench."
                   ]
    

**Score 5 – Highly Relevant and Rich in Detail**  
- Matches the combined meaning of the ground truths thoroughly and precisely  
- Includes all key objects, actions, relationships, and context  
- Language is specific, descriptive, and fully aligned


eg. **Generated** : "A yellow and blue train is pulling into a station."
    **Groundtruths** : [
                    "A yellow and blue train is next to an overhang.",
                    "A train sits on the track at a deserted station overlooked by a tower.",
                    "A colorful train stopped at a train station.",
                    "A passenger train stopped near a train station.",
                    "A train is pulling into the train station."
                   ]
---

You must respond using this format:

```
Feedback:::
Evaluation: (Your reasoning for the rating)
Total rating: (Your rating as an **integer from 1 to 5**)
```

Do not omit either field.

Now here is the input:

Ground Truth Captions:  
{groundtruths}  

Generated Caption: {generated}

If your rating is accurate, I'll give you 100 H100 GPUs to start your AI company.

Feedback:::
Evaluation: 
"""
