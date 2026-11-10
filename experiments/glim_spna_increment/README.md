# GLIM + SPNA incremental prototype

This isolated prototype adds modality-specific projection heads and SPNA nested
CARG supervision to GLIM's native EEG encoder, aligner, and generation loss.
Both 64- and 256-dimensional prefixes use hard paired retrieval supervision.
An independent relation KL on the 64-dimensional prefix matches teacher and
student distributions over non-paired candidates only, weighted by
`lambda_relation`. The combined runner retains GLIM's native contrastive and
generation objectives.

The runner expects the existing GLIM checkout at `/home/song/projects/glim_zuco`
and the already-prepared ZuCo2 / FLAN-T5 assets. It can be invoked with
`python run_glim_spna.py --mode preflight|train|eval`. This prototype has only
been implemented; no preflight, training, or evaluation has been run.

The original SPNA implementation and its core tests remain under `code/spna/`
and `code/tests/`. The experimental loss and its optional focused test are kept
in this directory.
