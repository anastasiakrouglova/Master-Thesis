# Thesis Anastasia Krouglova

Title: Music Analysis using Spectral Knowledge Representation and Reasoning.

My master's thesis uses FTP (Fast Padé Transform) to decompose audio signals into resonances and creates a multidimensional hierarchical structure with the help of machine learning. The philosophy behind the software architecture is a crucial part of this thesis. We conceptually step off from the mainstream approach towards data analysis and machine learning. The system I'm creating is based on abstract types, which makes it more expandable and flexible for various end-user goals. 

 Applications of the software could be:
- extracting instruments from a polyphonic audio signal
- converting music recordings into sheet scores
- ...



Run:

## Generating a .csv file of resonances

Run `execute_fpt.py`. This will generate
- a .csv file of the resonances + a f0 column in a form of a boolean (1 if resonace is part of fundamental, 0 if not)

## Clustering the fundamental resonances or harmnonics

2. Run Clustering_f0.jl
2. Run score_conversion/extract_fundamentals.jl with same filename
3. Run text_to_music.py if fails, otherwise just execute.py
4. Check new piano_score.pdf