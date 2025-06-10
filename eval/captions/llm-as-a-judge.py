JUDGE_PROMPT = """
You will be given a generated caption and a ground truth caption for a specific image.
Your task is to provide a 'total rating' scoring how well the generated caption relates to the ground truth captions.
Give your answer on a scale of 1 to 5, where 1 means that the generated caption is not relevant at all, and 5 means that the generated caption completely relates to the ground truth caption.

Here is the scale you should use to build your answer:

Score 1 – Completely Irrelevant and Vague  
- Caption does not relate to the image content at all.  
- May describe something unrelated or be nonsensical.  
- Lacks meaningful detail.

Score 2 – Minimally Relevant, Low Detail  
- Mentions a correct object or scene type, but overall content is inaccurate.  
- Omits key elements or relationships.  
- Uses generic language with limited specificity.

Score 3 – Moderately Relevant and Informative  
- Caption captures the general idea of the image.  
- Includes some objects, actions, or relationships, but may miss finer or secondary details.  
- Sufficient to understand the scene, but not comprehensive.

Score 4 – Mostly Relevant and Detailed  
- Accurately describes most of the important elements in the image.  
- Includes appropriate attributes, actions, and spatial relations.  
- Minor omissions or minor inaccuracies may exist, but overall the description is strong.

Score 5 – Highly Relevant and Rich in Detail  
- Fully captures the image content with precision.  
- Describes all key objects, actions, relationships, and context clearly.  
- Uses specific, descriptive language with no major omissions or errors.

Provide your feedback as follows:

Feedback:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 4)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and answer.

Question: {question}
Answer: {answer}

Provide your feedback. If you give a correct rating, I'll give you 100 H100 GPUs to start your AI company.
Feedback:::
Evaluation: """

