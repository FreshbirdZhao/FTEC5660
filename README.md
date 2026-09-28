# FTEC5660 Homework 1: Receipt Chain

Build a LangChain pipeline that reads every supermarket receipt in a folder
with the vision-capable DeepSeek Flash model and answers these two questions:

1. How much money did I spend in total for these bills?
2. How much would I have had to pay without the discount?

For this homework, **amount spent** means the final payment after the receipt's
rounding line. **Without the discount** means the sum of the original positive
item prices: add back every promotion, coupon, member, app, packaging-damage,
and percentage discount, but do not add back rounding.

## Student task

Only edit the two functions in `hw1.py` that contain `### YOUR CODE HERE`:

- `build_chain()` creates your LangChain chain.
- `answer_queries()` runs the chain on the receipt images and returns one final
response for each question.

You may use prompt chaining, routing, parallel calls, reflection, or a
combination. Your final responses should each contain one HKD amount. Do not
hard-code filenames or public answers; grading uses unseen receipt folders.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your DeepSeek key after `DEEPSEEK_API_KEY=` in `.env`, then run:

```bash
python3 hw1.py --image-folder public_test
```

The program creates `results.csv` in the current directory. Its columns are
`query`, `model_response`, and `correctness`. The public answers are in
`public_test/ground_truth.json`. The starter intentionally returns the dummy
response `please design your chain to answer these two queries.` so it runs
before you add any API code.

The required model is `deepseek-v4-flash-vision-exp`, the vision-capable
DeepSeek Flash model. JPEG, PNG, GIF, and WebP inputs are accepted by the
homework runner.

## Homework 1 solution

```mermaid
flowchart LR
    A[Receipt folder] --> B[Load and encode images]
    B --> C[Parallel DeepSeek vision calls]
    C --> D[Extract paid, subtotal, discounts]
    D --> E{JSON and arithmetic checks}
    E -->|invalid| F[Re-read image and retry]
    F --> D
    E -->|valid| G[Decimal aggregation]
    G --> H[Two exact HKD responses]
```



The chain sends each receipt separately to `deepseek-v4-flash-vision-exp` through LangChain so that the model only has to inspect one image at a time. A strict multimodal prompt extracts the final payment after rounding, the subtotal before rounding, and all discount lines into a four-field JSON object. The program validates the JSON, checks that the no-discount amount equals the subtotal plus discounts, checks that the payment is consistent with rounding, and asks the model to re-read any unreliable receipt. Finally, Python `Decimal` arithmetic aggregates the validated values and formats each required response as exactly one HKD amount.

