"""One model over two machines: the Mac keeps the embedding, layers [0, K), the head and the drafter; a stage runs the rest.

``tensorfold serve MODEL --split HOST --split-layers K`` on the Mac pushes the stage's layers to ``tensorfold stage`` on
the other machine (cached there by content hash), then sends each prompt chunk's and decode window's activations at
layer K and reads the final normed rows back.
"""
