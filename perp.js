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

// https://gist.github.com/engelen/fbce4476c9e68c52ff7e5c2da5c24a28
function argmax(array) {
    return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
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
    assert n >= 0, f"Invalid input {n} < 0"

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
    assert n >= 0, f"Invalid input {n} < 0"

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
        const target = Number(impTokens[i]);

        /** @type {string} */
        const token = tokenizer.decode([target]);
        const context = tokenizer(tokenizer.decode(sigTokens.concat(impTokens.slice(0, i))));
        // this is not perplexity
        // this calculates how much the model "agrees" with the implementation
        const logits = (await model(context)).logits[0][-1].data;
        const probs = softmax(logits);
        const best = argmax(Array.from(probs));
        const bestProb = probs[best];
        const bestToken = tokenizer.decode([best]);
        const targetProb = probs[target];
        /** @type {number} */
        const perplexity = targetProb / bestProb;
        if (perplexity !== 1.0) {
            console.log(`${i.toString().padStart(3, ' ')}/${impTokens.length} = ${perplexity.toPrecision(16)} '${token}' -> '${bestToken}'`);
        }
        scores.push(perplexity)
    }

    return scores;
}

async function main() {
    const pipe = await pipeline("text-generation", "Xenova/phi-1_5_dev");

    const goodP = await calculatePerplexity(pipe, good);
    const badP = await calculatePerplexity(pipe, bad);

    console.log([goodP, badP]);
    console.log(mean(goodP));
    console.log(mean(badP));
}

main();
