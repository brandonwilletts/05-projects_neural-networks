# Neural Network → LLM-from-Scratch Project Roadmap

## Purpose

This roadmap is designed to take you from your first PyTorch models to
the point where beginning Stanford CS336: *Language Modeling from
Scratch* is a reasonable next step. It deliberately avoids a broad tour
of every deep-learning application. Instead, each project introduces
capabilities that are directly useful for understanding and implementing
language models.

The progression is:

**basic optimization → perceptrons → MLPs → PyTorch abstractions → data
pipelines → language modeling → embeddings → attention → Transformers →
GPT → lower-level implementations → training systems**

The current CS336 course expects strong familiarity with PyTorch, deep
learning, Python/software engineering, and basic systems concepts. Its
first assignment already asks students to implement the components
needed for a Transformer language model, including the tokenizer, model
architecture, optimizer, and training loop. Later assignments move into
profiling, custom GPU kernels, distributed training, scaling, data
processing, and post-training. This roadmap is therefore intended as
preparation for the *starting line* of CS336, not a replacement for it.

## How to Use This Roadmap

For each project:

1.  Read only the project specification.
2.  Build the project yourself.
3.  Use documentation to look up API syntax.
4.  Avoid copying an existing implementation of the project.
5.  Run every required test.
6.  Do not move on until all **Required Completion Criteria** pass.
7.  Complete the **Understanding Check** without looking at your notes.

Looking up syntax such as `torch.matmul`, `nn.Module`, or `DataLoader`
is normal. Looking up an entire implementation of the project defeats
the purpose.

### General testing rules

Whenever a project uses **synthetic data**, the specification gives you the exact generation recipe. You implement that recipe in PyTorch, but you are not expected to invent the distribution, dimensions, labels, noise model, or train/test split.

Unless a project says otherwise:

-   Set `torch.manual_seed(42)` before initializing models or synthetic
    data.
-   Separate training and evaluation data.
-   Record training and validation loss.
-   Test tensor shapes explicitly.
-   Test at least one small input where you know the expected answer.
-   For numerical comparisons, use `torch.testing.assert_close`.
-   Save your final hyperparameters and results in the project's README.
-   When a threshold is defined relative to a baseline, calculate and
    record the baseline rather than hard-coding an assumed value.

A project is not complete merely because "the loss went down." The tests
below are intended to catch implementations that appear to work while
containing conceptual or implementation errors.

## Completion Tracker

-   [ ] Project 1 --- Linear Regression
-   [ ] Project 2 --- Perceptron
-   [ ] Project 3 --- Manual XOR MLP
-   [ ] Project 4 --- PyTorch XOR MLP
-   [ ] Project 5 --- MNIST MLP
-   [ ] Project 6 --- Bigram Language Model
-   [ ] Project 7 --- Contextual MLP Language Model
-   [ ] Project 8 --- Single-Head Self-Attention
-   [ ] Project 9 --- Multi-Head Self-Attention
-   [ ] Project 10 --- Transformer Block
-   [ ] Project 11 --- Tiny GPT
-   [ ] Project 12 --- Tiny GPT Training/Evaluation
-   [ ] Project 13 --- Core Components from Scratch
-   [ ] Project 14 --- Mini Training System
-   [ ] CS336 readiness gate passed
-   [ ] Project 15 --- Production-Style LLM Inference Service
-   [ ] Project 16 --- LLM Evaluation, Benchmarking, and Deployment
-   [ ] AI systems / portfolio readiness gate passed

------------------------------------------------------------------------

# Stage I --- Neural-Network Foundations

## Project 1 --- Linear Regression with Raw PyTorch Tensors

### Objective

Implement and train the model

`ŷ = wx + b`

using PyTorch tensors while keeping the learning process as explicit as
possible.

Do **not** use `nn.Linear`, `nn.Module`, or a PyTorch optimizer. You may
use PyTorch autograd to compute gradients.

The purpose is to connect the ideas of parameters, prediction, loss,
derivatives, and gradient descent directly to code.

### Dataset

Generate the dataset yourself so the correct parameters are known exactly. You are **not expected to invent how the dataset should be generated**; follow this recipe.

#### Exact data-generation recipe

1. Set the PyTorch random seed to `42`.
2. Generate `1,000` training `x` values independently from a uniform distribution over `[-5, 5]`.
3. Generate `1,000` independent Gaussian noise values with mean `0` and standard deviation `0.25`.
4. Calculate every training target using `y = 3x + 2 + noise`.
5. Generate the test set separately: `200` new `x` values from `[-5,5]`, `200` new Gaussian noise values with standard deviation `0.25`, and calculate each target with the same rule.
6. Do not reuse training observations in the test set.

To generate values in `[-5,5]`, start with uniform random values in `[0,1)`, multiply by `10`, and subtract `5`. For the noise, start with standard-normal random values and multiply by `0.25`.

The model must **never be given the rule `y = 3x + 2`**. It receives only the generated `(x,y)` observations. You know the hidden rule so you can test whether training recovers approximately `w = 3` and `b = 2`.

### Required implementation

Your program must:

-   Create trainable scalar tensors `w` and `b`.
-   Compute predictions directly using tensor operations.
-   Calculate mean squared error (MSE).
-   Use autograd to calculate `dL/dw` and `dL/db`.
-   Update `w` and `b` yourself using gradient descent.
-   Clear gradients correctly after every update.
-   Track training loss.
-   Evaluate on the held-out test set.

### Required Completion Criteria

All of the following must pass:

1.  **Parameter recovery**
    -   Final `w` must be between `2.95` and `3.05`.
    -   Final `b` must be between `1.90` and `2.10`.
2.  **Prediction test**
    -   With the trained model, the prediction for `x = 4.0` must fall
        between `13.7` and `14.3`.
    -   The noise-free correct value is `14`.
3.  **Loss test**
    -   Final training MSE must be below `0.10`.
    -   Test MSE must be below `0.10`.
4.  **Learning test**
    -   Final training loss must be less than 10% of the initial
        training loss.
5.  **Gradient test**
    -   Before training, compare your autograd gradient for `w` against
        a finite-difference approximation:
        `dL/dw ≈ [L(w + ε) - L(w - ε)] / (2ε)`
    -   Use `ε = 1e-4`.
    -   Relative error must be below `1e-3`.

### Understanding Check

Without looking at the code, explain:

-   What `w` and `b` represent.
-   Why MSE is a function of `w` and `b`.
-   What `w.grad` represents.
-   Why moving opposite the gradient should reduce the loss.
-   Why gradients must be cleared between optimization steps.

------------------------------------------------------------------------

## Project 2 --- Single Perceptron / Binary Classifier

### Objective

Implement a single-neuron binary classifier and train it on a linearly
separable two-dimensional dataset.

This project introduces classification, logits, probabilities, decision
boundaries, and binary cross-entropy.

Do not use `nn.Linear` or `nn.Module`.

### Dataset

Generate exactly 2,000 two-dimensional points. You are **not expected to invent the generation method**; follow this recipe.

#### Exact data-generation recipe

1. Set the PyTorch random seed to `42`.
2. Generate `2,000` independent `x1` values uniformly over `[-2,2]`.
3. Generate `2,000` independent `x2` values uniformly over `[-2,2]`.
4. Pair corresponding values into observations `(x1,x2)`.
5. Calculate `score = 2*x1 - x2 + 0.5` for each observation.
6. Assign `y = 1` when `score > 0`; otherwise assign `y = 0`.
7. Use observations `0–1599` for training and `1600–1999` for testing.
8. Do not add noise.

To generate values in `[-2,2]`, start with uniform random values in `[0,1)`, multiply by `4`, and subtract `2`. The classifier is not given the labeling rule; the known rule is only your reference for evaluating it.

### Required implementation

Implement:

`z = w1*x1 + w2*x2 + b`

followed by a sigmoid probability.

Train `w1`, `w2`, and `b` using gradient descent and binary
cross-entropy.

### Required Completion Criteria

1.  **Test accuracy**
    -   At least `99%` on the 400-point test set.
2.  **Known-point tests**
    -   `(2, 0)` must be classified as `1`.
    -   `(-2, 0)` must be classified as `0`.
    -   `(0, 1)` must be classified as `0`.
    -   `(0, -1)` must be classified as `1`.
3.  **Boundary-direction test**
    -   The learned weight vector should point in approximately the same
        direction as the true vector `[2, -1]`.
    -   Compute cosine similarity between `[w1, w2]` and `[2, -1]`.
    -   It must exceed `0.98`.
4.  **Loss test**
    -   Final training binary cross-entropy must be below `0.10`.
5.  **Probability sanity test**
    -   `P(y=1 | x=(2,0))` must exceed `0.90`.
    -   `P(y=1 | x=(-2,0))` must be below `0.10`.

### Understanding Check

Explain:

-   Why one neuron can solve this dataset.
-   What the learned weights geometrically represent.
-   What the bias does to the decision boundary.
-   Why sigmoid is useful for binary classification.
-   Why this model would fail on XOR.

------------------------------------------------------------------------

## Project 3 --- Two-Layer MLP from Explicit Parameters

### Objective

Build a multilayer perceptron capable of solving XOR.

This is the first project where a hidden layer and nonlinear activation
are essential.

Do not use `nn.Linear`. Define weight matrices and biases explicitly as
trainable tensors. You may use PyTorch autograd.

### Dataset

Use the complete XOR truth table:

    x1   x2   y
  ---- ---- ---
     0    0   0
     0    1   1
     1    0   1
     1    1   0

Because there are only four examples, train on all four.

### Required architecture

Use:

-   2 inputs
-   one hidden layer with at least 4 neurons
-   `tanh` or ReLU hidden activation
-   one output logit
-   sigmoid/BCE interpretation for binary classification

Implement the matrix operations yourself.

### Required Completion Criteria

1.  **Exact classification**
    -   All four XOR inputs must be classified correctly.
2.  **Probability confidence**
    -   `(0,0)` → probability `< 0.10`
    -   `(0,1)` → probability `> 0.90`
    -   `(1,0)` → probability `> 0.90`
    -   `(1,1)` → probability `< 0.10`
3.  **Loss**
    -   BCE across all four examples must be below `0.05`.
4.  **Shape tests** For a batch of 4:
    -   input shape: `(4, 2)`
    -   hidden preactivation: `(4, H)`
    -   hidden activation: `(4, H)`
    -   output logits: `(4, 1)`
5.  **Ablation test**
    -   Replace the hidden activation with the identity function and
        retrain several times.
    -   The resulting purely linear network must fail to achieve the
        same confident XOR solution.
    -   Record this result in the README.

### Understanding Check

Explain why stacking linear transformations without a nonlinear
activation is still equivalent to one linear transformation, and
therefore cannot solve XOR.

------------------------------------------------------------------------

## Project 4 --- Rebuild the XOR MLP with `nn.Module`

### Objective

Rebuild Project 3 using standard PyTorch abstractions.

The point is to see that `nn.Module`, `nn.Linear`, and an optimizer do
not change the underlying mathematics---they package operations you
previously performed manually.

### Dataset

Use the exact same XOR dataset as Project 3.

### Required implementation

Use:

-   a custom class inheriting from `nn.Module`
-   `nn.Linear`
-   a nonlinear activation
-   `forward()`
-   `torch.optim.SGD` or Adam
-   an appropriate binary classification loss

### Required Completion Criteria

1.  Pass the exact same XOR classification and confidence tests from
    Project 3.
2.  BCE below `0.05`.
3.  Print all parameters returned by `model.parameters()` and identify
    which correspond to the weights and biases you manually created in
    Project 3.
4.  Verify every trainable parameter has a non-`None` gradient after
    `loss.backward()`.
5.  Verify at least one parameter changes after `optimizer.step()`.
6.  Save the model with `state_dict()`, create a fresh model, reload the
    state dictionary, and verify the new model produces the same four
    logits to within absolute tolerance `1e-6`.

### Understanding Check

Be able to map every important line in the PyTorch implementation to the
corresponding operation in Project 3.

------------------------------------------------------------------------

# Stage II --- Real Data and Training Pipelines

## Project 5 --- MNIST MLP Classifier

### Objective

Train a fully connected neural network to recognize handwritten digits.

This introduces real datasets, minibatches, train/test separation,
`Dataset`, `DataLoader`, multiclass logits, and cross-entropy.

Do not use a convolutional neural network. The model must be an MLP.

### Dataset

Use MNIST through:

`torchvision.datasets.MNIST`

Official PyTorch documentation:

https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html

MNIST provides 28×28 grayscale digit images and predefined training/test
sets. Use the standard:

-   60,000 training images
-   10,000 test images

Normalize pixel values to `[0,1]`.

### Suggested architecture

A reasonable starting point:

`784 → 128 → ReLU → 64 → ReLU → 10`

You may modify hidden dimensions.

### Required Completion Criteria

1.  **Test accuracy**
    -   At least `97.0%` on the complete 10,000-image MNIST test set.
2.  **Training accuracy**
    -   At least `98.0%`.
3.  **Output shape**
    -   For batch size `B`, logits must have shape `(B, 10)`.
4.  **Loss interpretation**
    -   Feed the model one test image.
    -   Verify the predicted class is `argmax(logits)`.
5.  **Per-class evaluation**
    -   Calculate test accuracy independently for digits 0--9.
    -   Every digit must achieve at least `94%` accuracy.
6.  **Confusion analysis**
    -   Produce a 10×10 confusion matrix.
    -   Identify the three most common incorrect
        `(true digit → predicted digit)` pairs.
7.  **Generalization test**
    -   Test accuracy must be measured with the model in evaluation mode
        and without updating parameters.
8.  **Serialization test**
    -   Save and reload the model.
    -   Accuracy after reloading must be identical to accuracy before
        saving.

### Understanding Check

Explain:

-   Why the output layer has 10 logits.
-   Why those logits do not need to be probabilities before
    cross-entropy.
-   What a minibatch is.
-   Why training and test data must be separate.
-   What one epoch means.

------------------------------------------------------------------------

# Stage III --- Enter Language Modeling

## Project 6 --- Character-Level Bigram Language Model

### Objective

Build your first language model.

The model receives one character and predicts the next character. This
establishes the central language-modeling problem before adding longer
context or Transformers.

### Dataset

Use **Tiny Shakespeare**, the same small corpus popularized by
Karpathy's character-level examples.

Source:

https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

Reference preparation:

https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare_char/prepare.py

Expected dataset properties:

-   `1,115,394` characters
-   `65` unique characters
-   use the first 90% for training
-   use the final 10% for validation

Your program must derive the vocabulary from the file rather than
hard-code it.

### Required implementation

Create:

-   character-to-integer mapping
-   integer-to-character mapping
-   `encode(text)`
-   `decode(ids)`
-   training pairs `(character_t, character_t+1)`
-   a learnable 65×65 table of next-character logits, either directly or
    through an embedding-style lookup
-   cross-entropy training
-   autoregressive generation

### Required Completion Criteria

1.  **Dataset integrity**
    -   Total characters: `1,115,394`.
    -   Vocabulary size: `65`.
    -   Train split: `1,003,854` characters.
    -   Validation split: `111,540` characters.
2.  **Tokenizer round-trip**
    -   For at least 100 randomly selected substrings:
        `decode(encode(text)) == text`.
3.  **Random-model baseline**
    -   A uniform random predictor over 65 characters has cross-entropy
        `ln(65) ≈ 4.174`.
    -   Your trained model's validation loss must be below `3.0`.
4.  **Data-derived baseline**
    -   Build a unigram model from training-set character frequencies.
    -   Evaluate its validation cross-entropy.
    -   Your trained bigram model must beat that unigram validation
        loss.
5.  **Bigram sanity check**
    -   From the training corpus, calculate the empirical most-common
        next character after a newline.
    -   Query your trained model after a newline.
    -   Its top predictions should reflect the training distribution;
        specifically, the empirical most-common next character must
        appear in the model's top 5.
6.  **Generation**
    -   Generate at least 1,000 characters autoregressively.
    -   Every generated character must belong to the 65-character
        vocabulary.
    -   The output should contain spaces/newlines and recognizable
        word-like fragments rather than uniform random characters.

### Understanding Check

Explain why this is genuinely a language model even though it only sees
one previous character.

------------------------------------------------------------------------

## Project 7 --- Contextual MLP Language Model

### Objective

Improve on the bigram model by allowing the model to use multiple
previous characters.

This introduces embeddings, context windows, concatenated
representations, and the idea that predictions can depend on a sequence
rather than one token.

### Dataset

Use the exact same Tiny Shakespeare file, vocabulary, and 90/10 split
from Project 6.

Use a context length of at least `8` characters.

### Required implementation

For each training example:

-   input = previous `T` character IDs
-   target = next character ID

Create:

-   an embedding table
-   embeddings for all characters in the context
-   a combined/flattened context representation
-   at least one hidden MLP layer
-   output logits over all 65 characters

Do **not** use attention yet.

### Required Completion Criteria

1.  **Shape tests** With batch `B`, context length `T`, embedding
    dimension `C`:
    -   token IDs: `(B,T)`
    -   embeddings: `(B,T,C)`
    -   flattened MLP input: `(B,T*C)`
    -   logits: `(B,65)`
2.  **Bigram comparison**
    -   Evaluate Project 6 and Project 7 on exactly the same validation
        split.
    -   Project 7 validation cross-entropy must be at least `0.10` lower
        than Project 6.
3.  **Context dependence test**
    -   Choose two validation contexts ending in the same final
        character but with different preceding characters.
    -   The model must produce different next-character probability
        distributions.
    -   Verify they are not identical with `torch.allclose`.
4.  **Overfit test**
    -   Train on only 32 fixed context-target examples.
    -   Training accuracy on those 32 examples must reach at least
        `95%`.
5.  **Generation**
    -   Generate at least 1,000 characters.
    -   Compare a sample with Project 6 and record both validation
        losses in the README.

### Understanding Check

Explain what an embedding represents and why this model can use more
information than a bigram model but still has a rigid fixed context
structure.

------------------------------------------------------------------------

# Stage IV --- Build Attention

## Project 8 --- Single-Head Causal Self-Attention from Scratch

### Objective

Implement one causal self-attention head using basic PyTorch tensor
operations.

Do not use `nn.MultiheadAttention` or `scaled_dot_product_attention`.

The goal is to understand exactly how Q, K, V, attention scores, causal
masking, softmax, and weighted value aggregation work.

### Dataset

No external dataset is required. Follow this exact synthetic-data recipe: set seed `42`; set `B=2`, `T=4`, `C=8`, `H=4`; generate one `(2,4,8)` tensor whose values are independently drawn from a standard normal distribution; and keep that tensor fixed throughout the correctness tests. The values have no semantic meaning—they provide a controlled, reproducible input.

### Required implementation

Implement:

1.  linear projection to Q
2.  linear projection to K
3.  linear projection to V
4.  `Q @ Kᵀ`
5.  scaling by `sqrt(H)`
6.  causal mask
7.  row-wise softmax
8.  weighted sum of V

### Required Completion Criteria

1.  **Shape tests**
    -   Q, K, V: `(2,4,4)`
    -   attention scores: `(2,4,4)`
    -   attention probabilities: `(2,4,4)`
    -   output: `(2,4,4)`
2.  **Probability test**
    -   Every allowed attention row must sum to `1.0` within tolerance
        `1e-6`.
3.  **Causality test**
    -   Every attention probability above the main diagonal must be
        exactly zero (or numerically zero within `1e-7`).
4.  **First-token test**
    -   The first token can attend only to itself.
    -   Its attention vector must be `[1,0,0,0]` within tolerance
        `1e-6`.
5.  **Future-information invariance**
    -   Compute output for the original input.
    -   Change only token 4's input representation by adding a large
        value such as `100`.
    -   Recompute.
    -   Outputs at positions 1--3 must remain unchanged within `1e-5`.
    -   Position 4 may change.
6.  **Reference parity**
    -   Write a second, simple reference implementation of the same
        equations.
    -   With identical Q/K/V weights, outputs must match within
        `atol=1e-6`, `rtol=1e-5`.
7.  **Gradient test**
    -   Backpropagate the sum of the output.
    -   Input and Q/K/V projection parameters must receive finite
        gradients.

### Understanding Check

Be able to explain every dimension in the attention-score matrix and why
causal masking is required for next-token prediction.

------------------------------------------------------------------------

## Project 9 --- Multi-Head Causal Self-Attention

### Objective

Extend Project 8 to multiple attention heads.

The purpose is to understand that multiple heads allow separate
attention computations over different learned subspaces, followed by
recombination.

### Dataset

For correctness testing, follow this exact synthetic-data recipe: set seed `42`; set `B=2`, `T=8`, `d_model=32`, `n_heads=4`, `head_dim=8`; generate one `(2,8,32)` standard-normal tensor; and keep it fixed for all numerical tests. For the integration test, reuse the exact Tiny Shakespeare corpus, vocabulary construction, and 90/10 split from Project 6.

### Required implementation

Implement:

-   Q/K/V projections
-   split into 4 heads
-   causal attention independently per head
-   concatenate heads
-   final output projection

Do not use `nn.MultiheadAttention`.

### Required Completion Criteria

1.  **Shape tests**
    -   input: `(2,8,32)`
    -   split Q/K/V: `(2,4,8,8)` using your documented dimension order
    -   attention matrix per head: `(2,4,8,8)`
    -   concatenated result: `(2,8,32)`
    -   final result: `(2,8,32)`
2.  **Causal-mask test**
    -   For every batch and every head, all future attention weights
        must be zero within `1e-7`.
3.  **Probability test**
    -   For every head and query position, allowed attention
        probabilities sum to 1 within `1e-6`.
4.  **Head independence test**
    -   Alter the V projection parameters for exactly one head.
    -   Before the final output projection, outputs for the untouched
        heads must remain unchanged.
5.  **Reference test**
    -   Compare your module against an independently written reference
        implementation using the same weights.
    -   Outputs must match within `atol=1e-5`, `rtol=1e-4`.
6.  **Gradient test**
    -   Every Q/K/V and output projection parameter must receive finite
        gradients.
7.  **Tiny Shakespeare integration**
    -   Insert the module into a minimal next-character prediction
        network.
    -   Train on 128 fixed sequences.
    -   It must be capable of reducing training loss by at least 50%
        from its initial value.

### Understanding Check

Explain why splitting a 32-dimensional representation into four
8-dimensional heads is not the same as simply performing one
32-dimensional attention calculation.

------------------------------------------------------------------------

# Stage V --- Assemble a Transformer

## Project 10 --- One Complete Transformer Block

### Objective

Implement a decoder-style Transformer block by combining the components
you have already built.

The block should contain:

-   normalization
-   causal multi-head self-attention
-   residual connection
-   normalization
-   MLP
-   residual connection

Use a modern pre-normalization structure.

### Dataset

For unit tests, set seed `42`; set `B=2`, `T=16`, `d_model=64`, `n_heads=4`, and MLP hidden size `256`; generate one `(2,16,64)` standard-normal tensor; and keep it fixed for block-level correctness tests. For integration, reuse Tiny Shakespeare and the exact 90/10 split from Project 6, constructing fixed sequences only from the training portion.

### Required Completion Criteria

1.  **Shape preservation**
    -   Input `(2,16,64)` must produce output `(2,16,64)`.
2.  **Residual test**
    -   Temporarily force all attention and MLP output-projection
        weights/biases to zero.
    -   The block output must equal its input within `1e-6`.
3.  **Causality test**
    -   Change only the final input token.
    -   Outputs for all earlier positions must remain unchanged within
        `1e-5`.
4.  **Normalization test**
    -   Feed a nonconstant random tensor through each normalization
        layer independently.
    -   Verify all outputs are finite and dimensions are preserved.
5.  **Gradient-flow test**
    -   Backpropagate from the block output.
    -   Every trainable attention, normalization, and MLP parameter must
        have a finite gradient.
6.  **Tiny-dataset overfit**
    -   Put the block inside a minimal character LM with token
        embeddings and an output head.
    -   Use exactly 64 fixed Tiny Shakespeare sequences.
    -   Train until next-character training accuracy exceeds `95%` or
        training CE falls below `0.20`.

### Understanding Check

Explain the distinct job of:

-   attention
-   the MLP
-   normalization
-   residual connections

and why a Transformer repeatedly alternates attention and MLP
processing.

------------------------------------------------------------------------

# Stage VI --- Build GPT

## Project 11 --- Tiny GPT from Scratch

### Objective

Build a complete decoder-only GPT-style language model by stacking your
Transformer blocks.

This is the major synthesis project.

### Dataset

Use Tiny Shakespeare with the same 90/10 character-level split.

Recommended initial configuration:

-   vocabulary: `65`
-   context length: `128`
-   `d_model`: `128`
-   heads: `4`
-   layers: `4`
-   MLP expansion: `4×`
-   learned token embeddings
-   learned positional embeddings
-   final normalization
-   linear language-model head

You may reduce dimensions temporarily for CPU debugging.

### Required implementation

Your model must perform:

`token IDs → token embeddings + position embeddings → N Transformer blocks → final norm → vocabulary logits`

Train with next-character cross-entropy.

### Required Completion Criteria

1.  **Output shape**
    -   Input `(B,T)` → logits `(B,T,65)`.
2.  **Parameter sanity**
    -   Print total trainable parameter count.
    -   Independently calculate the expected parameter count from the
        architecture.
    -   Counts must match exactly.
3.  **Causality**
    -   Changing token `t+1` must not change logits at positions `≤ t`
        within `1e-5`.
4.  **Tiny-batch overfit**
    -   Select 16 fixed training sequences.
    -   Your GPT must drive training CE below `0.10` on those fixed
        sequences.
    -   Do this before attempting full training.
5.  **Full training**
    -   Train on the full Tiny Shakespeare training split.
    -   Validation cross-entropy must fall below `2.5`.
    -   Stretch goal: below `2.2`.
6.  **Baseline superiority**
    -   It must beat the validation losses recorded for Projects 6 and
        7.
7.  **Generation**
    -   Generate at least 2,000 characters from a short prompt.
    -   All generated token IDs must be valid.
    -   Generation must be autoregressive: each newly generated token is
        appended and used in subsequent predictions.
8.  **Train/eval separation**
    -   Record both train and validation loss at regular intervals.
    -   Validation evaluation must not update parameters.

### Understanding Check

Starting with one integer token ID, verbally trace its path through the
entire model until the final probability distribution for the next
character.

------------------------------------------------------------------------

## Project 12 --- Train and Evaluate the Tiny GPT Properly

### Objective

Turn Project 11 from "a model that trains" into a controlled
language-model experiment.

The focus shifts from architecture construction to training discipline,
evaluation, reproducibility, sampling, and diagnosis.

### Dataset

Continue using Tiny Shakespeare.

Keep the same fixed 90/10 split so results are directly comparable with
Projects 6, 7, and 11.

### Required additions

Implement:

-   reproducible random seeds
-   random minibatch sampling
-   periodic validation evaluation
-   training/validation loss logging
-   checkpoint saving
-   checkpoint loading
-   greedy generation
-   temperature sampling
-   top-k sampling
-   gradient norm monitoring

### Required Completion Criteria

1.  **Reproducibility**
    -   Run the first 20 training steps twice from the same seed and
        configuration.
    -   The loss sequence must match within `1e-5` on the same
        hardware/backend.
2.  **Checkpoint resume**
    -   Train to step 500 and save.
    -   Reload the checkpoint.
    -   Before any further update, validation loss must match the
        pre-save validation loss within `1e-5`.
3.  **Training continuation**
    -   Continue from the checkpoint.
    -   Validation loss must eventually improve beyond the value at
        initialization.
4.  **Validation target**
    -   Full validation CE below `2.5`.
    -   Stretch goal: `< 2.2`.
5.  **Sampling tests** Using the same prompt:
    -   temperature approaching zero should produce increasingly
        deterministic output.
    -   temperature `1.0` should produce variable samples when the RNG
        seed changes.
    -   top-k `k=1` must produce the same next-token choice as greedy
        decoding.
6.  **Context-window test**
    -   Generate beyond the model's context length.
    -   Verify the model receives no more than its configured maximum
        context at each generation step.
7.  **No-leakage test**
    -   Confirm that no validation characters are included in the
        training split.
    -   Record exact split indices in the README.
8.  **Learning curves**
    -   Produce a graph of training and validation loss against
        optimization steps.
    -   Explain whether the model is underfitting, fitting, or beginning
        to overfit.

### Understanding Check

Explain why validation loss is a more useful measure of language-model
generalization than judging a few generated samples by eye.

------------------------------------------------------------------------

# Stage VII --- Remove the Training Wheels

## Project 13 --- Reimplement Core Components and Prove Numerical Parity

### Objective

Reimplement important neural-network components rather than relying on
high-level PyTorch versions, then prove that your implementations agree
numerically with trusted reference operations.

This project moves you toward the implementation style expected by
CS336.

### Dataset

No large dataset is needed for the core tests. Set seed `42`, then generate these tensors in this exact order: `linear_input` `(8,16)` standard-normal; `normalization_input` `(4,8,32)` standard-normal; `classification_logits` `(32,65)` standard-normal; `classification_targets` as 32 integer IDs sampled from `{0,...,64}`; and `attention_input` `(2,8,32)` standard-normal. Use the same tensors for your implementation and its reference comparison. For final integration, reuse Tiny Shakespeare and the exact Project 6 split.

### Components to implement

Implement your own versions of at least:

1.  Linear layer
2.  ReLU or GELU
3.  softmax
4.  cross-entropy loss
5.  LayerNorm or RMSNorm
6.  causal self-attention
7.  Adam or AdamW optimizer

You may use basic PyTorch tensor operations and autograd unless the
component being tested specifically requires otherwise.

Do not call the high-level PyTorch equivalent inside your
implementation.

### Fixed numerical test inputs

Use seed `42` and include:

-   matrix input: `torch.randn(8,16)`
-   normalization input: `torch.randn(4,8,32)`
-   classification logits: `torch.randn(32,65)`
-   integer targets: 32 valid class IDs
-   attention input: `torch.randn(2,8,32)`

### Required Completion Criteria

1.  **Linear parity**
    -   Copy identical weights/biases into your layer and `nn.Linear`.
    -   Forward outputs: `atol=1e-6`, `rtol=1e-5`.
    -   Input and parameter gradients: `atol=1e-6`, `rtol=1e-5`.
2.  **Activation parity**
    -   Compare against PyTorch's corresponding activation.
    -   Forward and backward results within `1e-6` absolute tolerance
        where numerically appropriate.
3.  **Softmax**
    -   Rows sum to 1 within `1e-6`.
    -   Compare with `torch.softmax` within `1e-6`.
    -   Must remain finite on logits containing values such as
        `[1000, 1001, 999]`.
    -   Your implementation must therefore use a numerically stable
        formulation.
4.  **Cross-entropy**
    -   Compare with `torch.nn.functional.cross_entropy`.
    -   Loss difference `< 1e-6`.
    -   Logit gradients agree within `atol=1e-6`, `rtol=1e-5`.
5.  **Normalization**
    -   Compare against the matching PyTorch/reference formula with
        identical learned parameters.
    -   Forward output difference `< 1e-5`.
    -   Gradients agree within `1e-5`.
6.  **Attention**
    -   Must pass all causality/probability tests from Projects 8--9.
    -   Reference outputs within `1e-5`.
7.  **Optimizer**
    -   Initialize two identical tiny models.
    -   Train one for 10 deterministic steps with your Adam/AdamW and
        the other with PyTorch's optimizer using matching
        hyperparameters and semantics.
    -   Parameter tensors after every step must agree within a tolerance
        you document; target `atol <= 1e-6` for a strictly matched
        implementation.
8.  **Integration**
    -   Replace the corresponding components in your Tiny GPT with your
        own versions.
    -   Run one forward/backward step using identical model weights and
        a fixed Tiny Shakespeare batch.
    -   Loss must agree with the reference model within `1e-5`.

### Understanding Check

For every component, explain both:

1.  the mathematical operation it performs; and
2.  what convenience the PyTorch implementation had previously been
    providing.

------------------------------------------------------------------------

# Stage VIII --- Build a Small Training System

## Project 14 --- Mini LLM Training Infrastructure

### Objective

Turn your Tiny GPT implementation into a reliable training program
rather than a notebook experiment.

This project prepares you for the systems mindset of CS336, where
correctness is only the beginning: you must also reason about memory,
throughput, checkpoints, profiling, and reproducibility.

### Dataset

Use Tiny Shakespeare first because it is small enough for rapid
debugging.

After all correctness tests pass, optionally run the system on a larger
text dataset. Do not make the larger dataset a prerequisite for
completing the project.

### Required features

Your training program must support:

-   configuration of model hyperparameters
-   configuration of training hyperparameters
-   CPU and available accelerator selection
-   train/validation split loading
-   deterministic seed
-   minibatching
-   gradient descent with AdamW
-   gradient clipping
-   periodic evaluation
-   checkpoint save
-   checkpoint resume
-   best-validation checkpoint
-   training logs
-   tokens/second measurement
-   parameter count
-   approximate training-token count
-   optional mixed precision when supported
-   command-line or configuration-file control

### Required Completion Criteria

#### A. Checkpoint correctness

Perform this controlled experiment:

**Run A** - initialize from seed `42` - train continuously for 200
steps - save final parameters and loss history

**Run B** - initialize from seed `42` - train 100 steps - checkpoint -
terminate the process - reload - train steps 101--200

When RNG state, model state, optimizer state, and data-sampling state
are all restored correctly:

-   final loss should match Run A within `1e-5` on the same
    deterministic backend
-   corresponding final parameters should match within `1e-5`

If your hardware/backend cannot guarantee deterministic kernels,
document the limitation and require close numerical agreement plus
matching qualitative trajectory.

#### B. Tiny Shakespeare performance

Using the Tiny GPT family from Projects 11--12:

-   validation CE `< 2.5`
-   stretch goal `< 2.2`
-   checkpoint reload must reproduce pre-save validation loss within
    `1e-5`

#### C. Throughput measurement

Measure training throughput in **tokens per second**.

Run at least:

-   context length 64
-   context length 128
-   context length 256

using the same model dimensions and batch size where memory permits.

Record:

-   tokens processed
-   wall-clock time
-   tokens/sec
-   device

Do not require a particular speed; the test is that the measurement is
correct and reproducible to within roughly 15% across repeated
steady-state runs.

#### D. Memory experiment

For at least three batch sizes, record peak device memory where the
platform exposes it.

The reported memory usage should generally increase as batch size
increases. Investigate any contrary result rather than silently
accepting it.

#### E. Gradient health

During a 500-step run:

-   record global gradient norm
-   no loss may become NaN or infinity
-   no gradient norm may become NaN or infinity
-   demonstrate that configured gradient clipping actually caps the norm
    when an intentionally low clipping threshold is used

#### F. Best-checkpoint behavior

Whenever validation loss reaches a new minimum, save a "best"
checkpoint.

At the end:

-   reload the best checkpoint
-   recompute validation loss
-   it must match the recorded best validation loss within `1e-5`

#### G. Configuration reproducibility

A saved run configuration must contain enough information to recreate:

-   architecture
-   context length
-   optimizer settings
-   learning rate
-   batch size
-   seed
-   dataset/split
-   number of steps

A fresh process loading the same model checkpoint and configuration must
reproduce its validation result.

#### H. Failure/recovery test

Deliberately terminate a training run after a checkpoint has been
written.

Restart from the checkpoint and verify that training continues without
resetting:

-   model weights
-   optimizer state
-   step number
-   best validation score

### Understanding Check

Explain:

-   why optimizer state must be checkpointed in addition to weights;
-   what tokens/second measures;
-   why GPU memory can constrain batch size/context length;
-   why mixed precision can increase training efficiency;
-   why correctness should be established on CPU/small inputs before
    performance optimization.

------------------------------------------------------------------------


# Stage IX --- Turn the Model into an AI System

The first 14 projects deliberately prepare you to *begin* CS336. Complete
this stage after Project 14 and either during or after CS336. It is not
intended to duplicate the separate backend-project roadmap. Instead, it
adds the ML-specific production work that a generic backend project does
not normally test: model serving, inference benchmarking, evaluation,
experiment tracking, and deployment of a trained model.

The goal is to turn "I implemented and trained a language model" into
evidence that you can build, measure, test, and operate an AI system.

## Project 15 --- Production-Style LLM Inference Service

### Objective

Package a trained language model behind a clean inference interface and
treat inference as an engineering workload rather than a notebook demo.

Use the best checkpoint from Project 14 or a later CS336 model. The model
itself is not the focus of this project; serving behavior, correctness,
measurement, and reliability are.

If your separate backend roadmap already covers generic API construction,
Docker, logging, and deployment, reuse those skills here rather than
relearning them. The new requirement is to apply them to a real model.

### Required implementation

Build a small inference service that supports:

- loading a model and tokenizer once at process startup;
- a generation endpoint accepting a prompt and generation parameters;
- configurable `max_new_tokens`, temperature, and top-k;
- deterministic generation when a seed and deterministic sampling mode
  permit it;
- input validation and useful error responses;
- a health/readiness endpoint;
- structured request logging without logging full user prompts by default;
- CPU and available accelerator execution;
- Docker packaging;
- automated unit and integration tests.

Keep the API thin. Model loading, tokenization, generation, and request
handling should be separate modules.

### Required Completion Criteria

1. **Model-load correctness**
   - Load a known checkpoint and tokenizer.
   - For a fixed prompt, seed, and generation configuration, direct
     in-process generation and generation through the service must return
     identical token IDs.

2. **Tokenizer round-trip**
   - Test at least 100 representative strings.
   - For every string supported losslessly by the tokenizer,
     `decode(encode(text)) == text`.

3. **API correctness**
   - Valid requests return generated text and generation metadata.
   - Empty or malformed requests return a 4xx response rather than
     crashing the process.
   - Requests exceeding the configured context/generation limit are
     rejected or truncated according to a documented policy.

4. **Model reuse**
   - Instrument model initialization.
   - Across 100 sequential generation requests, the checkpoint must be
     loaded exactly once per service process.

5. **Concurrency sanity**
   - Send at least 20 concurrent requests.
   - Every request must complete successfully or return a documented
     capacity/rate-limit response.
   - The process must not crash, leak unbounded memory, or reload the
     model for each request.

6. **Automated tests**
   - Unit-test tokenization, validation, generation configuration, and at
     least one deterministic model output.
   - Add an integration test that starts the service and makes a real
     request through the API.
   - The complete test suite must run from one documented command.

7. **Container test**
   - Build the Docker image from a clean checkout.
   - Start the container with one documented command.
   - The health endpoint must become ready and the integration generation
     request must pass.

### Understanding Check

Explain:

- why a model should normally be loaded once rather than per request;
- the difference between training throughput and inference latency;
- why tokenization belongs in the serving pipeline;
- what can make generation nondeterministic;
- why an AI endpoint needs tests beyond "the response looks reasonable."

------------------------------------------------------------------------

## Project 16 --- LLM Evaluation, Benchmarking, and Deployment

### Objective

Build a reproducible evaluation and performance harness around the
inference service, deploy it, and produce evidence about model quality and
systems behavior.

This project is intentionally more important for a job portfolio than
adding another toy model. It should produce a concise engineering report
showing what you measured, how you measured it, what failed, and what you
changed.

### Dataset

Create a version-controlled evaluation set containing at least 200
prompt/target examples that your model can meaningfully score.

For a character- or token-level language model, construct these examples
from a held-out portion of the corpus that was never used for training or
hyperparameter selection. Store the exact source/split information needed
to recreate the set.

If you complete CS336 with a more capable tokenizer/model, replace or
extend this evaluation set with a task-appropriate held-out set while
preserving the same evaluation discipline.

### Required implementation

Implement:

- a reproducible offline evaluation command;
- held-out cross-entropy/perplexity measurement;
- comparison against at least one simpler baseline from the roadmap;
- latency measurement;
- throughput measurement;
- warm-up before performance measurements;
- batch-size experiments;
- prompt/context-length experiments;
- CPU versus accelerator comparison when both are available;
- peak-memory measurement when the platform exposes it;
- experiment results written to machine-readable output such as JSON or
  CSV;
- deployment of the containerized service to a real compute environment;
- a README/report explaining architecture, tests, metrics, bottlenecks,
  limitations, and one optimization experiment.

### Required Completion Criteria

1. **Evaluation reproducibility**
   - Run the complete evaluation twice against the same checkpoint and
     evaluation set.
   - Deterministic metrics such as loss/perplexity must agree within
     `1e-5` on the same deterministic backend.

2. **No evaluation leakage**
   - Programmatically verify that evaluation examples do not overlap the
     training split used for the reported model.
   - Record the dataset version/hash and split boundaries.

3. **Baseline comparison**
   - Evaluate the final model and at least one earlier baseline on exactly
     the same evaluation data.
   - The final model must outperform the baseline on held-out
     cross-entropy.

4. **Latency benchmark**
   - After warm-up, run at least 100 generation requests for a fixed
     prompt/context and generation length.
   - Report median (p50) and p95 end-to-end latency.
   - Repeat the benchmark at least three times; each run's p50 should be
     within 20% of the median p50 across runs, or the variance must be
     investigated and documented.

5. **Throughput benchmark**
   - Report generated tokens/second for at least three batch sizes or
     concurrency levels.
   - Identify the highest-throughput configuration that fits in available
     memory.

6. **Context-length experiment**
   - Benchmark at least three input context lengths while holding the
     generation length constant.
   - Record latency, throughput, and peak memory where available.
   - Explain the observed trend using the computational behavior of
     Transformer attention.

7. **Optimization experiment**
   - Make one measurable inference optimization, such as batching,
     mixed precision, compilation, or another optimization appropriate to
     your hardware/model.
   - Benchmark before and after under identical conditions.
   - Report the percentage change in latency and/or throughput.
   - Keep the optimization only if correctness tests continue to pass.

8. **Deployment test**
   - Deploy the containerized service to a real compute environment.
   - From a separate client process, verify the health endpoint and at
     least 20 generation requests.
   - Restart the deployed service and verify that it returns to a ready
     state and successfully reloads the configured checkpoint.

9. **Failure behavior**
   - Test at least: malformed input, context too long, generation request
     too large, and unavailable/corrupt checkpoint at startup.
   - Each failure must produce a documented error or failed-readiness
     state rather than silent incorrect behavior.

10. **Engineering report**
    - Include architecture/data-flow diagram.
    - Record model/checkpoint, tokenizer, dataset version, hardware,
      software versions, and benchmark configuration.
    - Include a table of quality metrics and a table of systems metrics.
    - Document at least three limitations or next steps.

### Understanding Check

Explain:

- why model quality and serving performance require separate metrics;
- the difference between p50 and p95 latency;
- why benchmarks require warm-up and controlled configurations;
- why throughput can improve while single-request latency worsens;
- why an evaluation set must be isolated from training and model
  selection;
- how you would decide whether an inference optimization is worth its
  complexity.

------------------------------------------------------------------------

# Final Readiness Gate --- Before Starting CS336

Completing all 14 projects does not mean you already know the material
in CS336. It means you should have removed many of the implementation
barriers that would otherwise make the course unnecessarily difficult.

Before starting CS336, you should be able to complete the following
without following a tutorial.

## PyTorch

You can:

-   create and manipulate multidimensional tensors;
-   reason about tensor shapes;
-   use broadcasting and matrix multiplication;
-   understand autograd;
-   write `nn.Module` classes;
-   write a training loop;
-   use optimizers;
-   use datasets and minibatches;
-   move models/data between CPU and accelerator devices;
-   save and restore model and optimizer state.

## Neural Networks

You can explain and implement:

-   linear models;
-   perceptrons;
-   MLPs;
-   activations;
-   logits;
-   cross-entropy;
-   gradient descent;
-   backpropagation conceptually;
-   embeddings;
-   normalization;
-   residual connections.

## Transformers

You can implement and explain:

-   Q/K/V projections;
-   scaled dot-product attention;
-   causal masking;
-   multi-head attention;
-   Transformer MLPs;
-   residual streams;
-   normalization;
-   positional information;
-   stacked Transformer blocks;
-   next-token prediction.

## Language Modeling

You can:

-   tokenize text at least at the character level;
-   construct context/target pairs;
-   calculate next-token cross-entropy;
-   train/validation split a corpus;
-   evaluate validation loss;
-   generate autoregressively;
-   use temperature and top-k sampling;
-   overfit a tiny dataset as a debugging test.

## Software / Systems

You can:

-   structure the model, data, training, and evaluation code cleanly;
-   write unit tests;
-   compare an implementation against a numerical reference;
-   use checkpoints;
-   reproduce an experiment;
-   measure throughput;
-   reason about tensor memory and computational cost;
-   debug shape and gradient problems.

## AI Systems / Portfolio Readiness

Before using this roadmap as evidence for AI-oriented SWE applications,
you can also:

-   serve a trained model behind a tested API;
-   package the service in Docker;
-   deploy it to a real compute environment;
-   build a reproducible held-out evaluation harness;
-   compare model quality against a baseline;
-   measure p50/p95 inference latency and tokens/second;
-   benchmark batch size, context length, and memory behavior;
-   make and validate at least one inference optimization;
-   distinguish model-quality failures from systems-performance failures;
-   document architecture, metrics, limitations, and engineering tradeoffs.

If several of these still feel mysterious, revisit the relevant project
rather than adding more tutorials.

------------------------------------------------------------------------

# Recommended Repository Structure

You can keep the entire sequence in one repository:

``` text
neural-networks-to-llm/
├── 01-linear-regression/
├── 02-perceptron/
├── 03-mlp-xor-manual/
├── 04-mlp-xor-pytorch/
├── 05-mnist-mlp/
├── 06-bigram-lm/
├── 07-context-mlp-lm/
├── 08-single-head-attention/
├── 09-multi-head-attention/
├── 10-transformer-block/
├── 11-tiny-gpt/
├── 12-tiny-gpt-training/
├── 13-components-from-scratch/
├── 14-training-system/
├── 15-inference-service/
└── 16-evaluation-benchmark-deploy/
```

Each project should ideally contain:

``` text
README.md
src/
tests/
results/
```

The README should record:

-   objective;
-   architecture;
-   dataset;
-   hyperparameters;
-   required test results;
-   final train/validation metrics;
-   what you learned;
-   remaining questions.

------------------------------------------------------------------------

# Recommended Rule for Getting Help

To avoid tutorial hell, use a hierarchy:

**Level 1 --- Documentation:** Look up syntax/API behavior.

**Level 2 --- Conceptual hint:** Ask what concept might be wrong without
asking for code.

**Level 3 --- Debugging hint:** Provide your code/error and ask where to
investigate.

**Level 4 --- Partial example:** Ask for a minimal example of the
specific operation you do not understand.

**Level 5 --- Full solution:** Use only after you have completed the
project or are completely blocked and have already attempted to diagnose
it.

For these projects, being able to struggle productively with an
implementation is part of the training.

------------------------------------------------------------------------

# Data and Reference Sources

## MNIST

PyTorch/Torchvision MNIST dataset documentation:

https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html

## Tiny Shakespeare

Raw corpus:

https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

Karpathy nanoGPT character-level preparation script:

https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare_char/prepare.py

Karpathy's bigram/GPT lecture implementation can be useful **after** you
have attempted the corresponding projects yourself:

https://github.com/karpathy/ng-video-lecture

## Stanford CS336

Current course:

https://cs336.stanford.edu/

CS336 describes itself as an implementation-heavy course covering the
complete language-model development process. As of Spring 2026,
Assignment 1 includes implementing the tokenizer, Transformer
architecture, optimizer, and minimal LM; later assignments cover
profiling and FlashAttention/Triton, distributed training, scaling,
pretraining data, and alignment/reasoning.
