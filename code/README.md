


## Generating a .csv file of Resonances

Run `execute_fpt.py`. This will generate a .csv file of the resonances with an additional column `f0`, generated using the Rameau Fundamental.

## Clustering the Fundamental Resonances or Harmonics

Run Clustering_f0.jl

## Creating a Musical Score

1. Run `score_conversion/extract_fundamentals.jl` with filename adjusted to the clustered output
2. Run `score_conversion/execute.py`text_to_music.py`. If the generation fails: download Lilypond and adjust the paths 
3. Check generated score.pdf


## Hierarchical Knowledge Representation

Test and expand the type-based hierarchical knowledge representation in `./resonance-knowledge`

