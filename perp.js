import { pipeline, softmax } from "@xenova/transformers";

function median(numbers) {
    const sorted = Array.from(numbers).sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);

    if (sorted.length % 2 === 0) {
        return (sorted[middle - 1] + sorted[middle]) / 2;
    }

    return sorted[middle];
}

function mean(numbers) {
    return numbers.reduce((a, b) => a + b) / numbers.length;
}

const bad = [
    `def fib(n):
    `,
    `if n == 0:
        return 0

    if n <= 1:
        return 1

    return n + fibonacci(n-1, n-2)
`
];

const good = [
    `def fibonacci(n: int):
    """
    Calculate the n-th Fibonacci number.

    Args:
        n (int) - nth number to calculate

    Returns:
        int - the nth fibonacci number.
    """
    `,
    `
    assert n >= 0, f"Invalid value {n=!r} < 0"

    if n <= 1:
        return 1

    return n + fibonacci(n-1, n-2)`
];

/**
import torch
from tqdm import tqdm

max_length = model.config.n_positions
stride = 512
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
 */

async function calculatePerplexity(pipe, [signature, implementation]) {
    const model = pipe.model;
    const tokenizer = pipe.tokenizer;

    const sigTokens = Array.from(tokenizer(signature).input_ids.data);
    const impTokens = Array.from(tokenizer(implementation).input_ids.data);

    const scores = [];
    for (let i = 1; i < impTokens.length - 1; i++) {
        const token = tokenizer.decode([impTokens[i]]);
        const context = tokenizer(tokenizer.decode(sigTokens.concat(impTokens.slice(0, i))));
        const target = Number(impTokens[i]);
        // this is wrong
        const perplexity = softmax((await model(context)).logits[0][-1].data)[target];
        console.log(`${i}/${impTokens.length}: '${token}' ${perplexity}`);
        scores.push(perplexity)
    }

    return scores;
}

async function main() {
    const pipe = await pipeline("text-generation", "Xenova/phi-1_5_dev");

    const goodP = await calculatePerplexity(pipe, good);
    const badP = await calculatePerplexity(pipe, bad);

    console.log([goodP, badP]);
    console.log([mean(goodP), median(goodP)]);
    console.log([mean(badP), median(badP)]);
}

main();
