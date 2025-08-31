# Design Choices

### Rendering Words

Words are rendered as they are, on a single line of text.
Control tokens are either invisible, or create issues in rendering (e.g. newlines).
Instead, we replace them with visible characters: https://unicode.org/charts/nameslist/n_2400.html

### Decoding Latent Vectors

In inference, we really just need to decode the latent vectors into bytes of the current "word".
However, since words are of different lengths, we need to pad the sequences to the maximum word length.
This is a bit inefficient, as we could use that padding for other computation, and so for now, instead of padding,
we let the model predict the next N bytes, where N is the maximum word length.
For example, from `["Hello"]`, assuming max word length of 10 it is trained to
predict `["w","o","r","l","d"," ","m","y"," ","n"]`.
This can be useful for speculative-decoding kind of approach.

### Bytes Decoder

Originally, we would have liked to use cross-attention in the bytes decoder,
however, not all causal language models support it.
Thus, we implemented a parallel causal decoding mechanism that prepends word-level latent vectors to the character
sequences, allowing us to use the self-attention mechanism instead.

At first, we added all previous latent vectors to the current character sequence,
however, that proved difficult to properly implement in a parallel manner.
This is because we would also concatenate padding tokens, and the attention mask should change accordingly,
as well as the position embeddings.
