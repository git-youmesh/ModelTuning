"""The underlying idea of TSDAE is that we add noise to the input sentence by
removing a certain percentage of words from it. This “damaged” sentence is
put through an encoder, with a pooling layer on top of it, to map it to a
sentence embedding. From this sentence embedding, a decoder tries to
reconstruct the original sentence from the “damaged” sentence but without
the artificial noise. The main concept here is that the more accurate the
sentence embedding is, the more accurate the reconstructed sentence will
be"""
