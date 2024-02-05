# Chess Position Evaluator 
 This is a chess position evaluator employing a deep learning neural network (deep NN).

## Encoding

The chessboard is encoded in the following manner: each square has 7 single-bit values associated with it. These values are:

- position = 0: Is the square **empty**?
- position = 1: Is there a **pawn**?
- position = 2: Is there a **knight**?
- position = 3: Is there a **bishop**?
- position = 4: Is there a **rook**?
- position = 5: Is there a **queen**?
- position = 6: Is there a **king**?


These 7 values make up a binary vector. For example, `[0 1 0 0 0 0 0 0]` represents the square where a pawn is present.

Notice that only one value can be `1` at once. This type of encoding is called *one-hot* encoding.

To represent entire chessboard, 64 such vectors are required.
Chessboard is represented rank (what in the chess terminology means `row`) by rank from left to right starting from A1.

In other words, the first vector corresponds to A1, second - B1, third - C1 and so on to H1. Then the next rank is encoded: A2, B2, C2, ..., H2. The process repeats until all 64 squares are encoded. In total is 64 * 7 = 448 values.

Moreover, a single value is require to encode who has the next move. This information is represented as a logical value, where `true` means that it is White's turn and `false` - Black's turn.

Therefore, the total input size is 449. All these values are stacked in a column vector. It makes the input layer look like:
$[0 0 0 0 0 0 1 0 0 0 0 1 0 0 ... 0 1]^T$