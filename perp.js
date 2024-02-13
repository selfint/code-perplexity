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
    `def fibonacci(n: int) -> int:
    """
    Calculate the n-th Fibonacci number.

    Args:
        n (int) - nth number to calculate.

    Returns:
        int - the nth fibonacci number.
    """
    `,
    `
    assert n >= 0, f"Invalid value {n=!r} < 0"

    if n <= 1:
        return n

    return n + fibonacci(n-1, n-2)`
];

const good = [
    `def fibonacci(n: int) -> int:
    """
    Calculate the n-th Fibonacci number.

    Args:
        n (int) - nth number to calculate.

    Raises:
        AssertionError - if n < 0.

    Returns:
        int - the nth fibonacci number.
    """
    `,
    `
    assert n >= 0, f"Invalid value {n=!r} < 0"

    if n <= 1:
        return n

    return n + fibonacci(n-1, n-2)`
];


async function calculatePerplexity(pipe, [signature, implementation]) {
    const model = pipe.model;
    const tokenizer = pipe.tokenizer;

    const sigTokens = Array.from(tokenizer(signature).input_ids.data);
    const impTokens = Array.from(tokenizer(implementation).input_ids.data);

    const scores = [];
    for (let i = 1; i < impTokens.length - 1; i++) {
        /** @type {string} */
        const token = tokenizer.decode([impTokens[i]]);
        const context = tokenizer(tokenizer.decode(sigTokens.concat(impTokens.slice(0, i))));
        const target = Number(impTokens[i]);
        // this is wrong
        /** @type {number} */
        const perplexity = softmax((await model(context)).logits[0][-1].data)[target];
        console.log(`${i.toString().padStart(3, ' ')}/${impTokens.length} = ${perplexity.toString().padStart(18, ' ')} '${token}'`);
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
