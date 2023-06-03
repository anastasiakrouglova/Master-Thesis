


## Generating a .csv file of resonances

Run `execute_fpt.py`. This will generate
- a .csv file of the resonances + a f0 column in a form of a boolean (1 if resonace is part of fundamental, 0 if not)

## Clustering the fundamental resonances or harmnonics

2. Run Clustering_f0.jl
2. Run score_conversion/extract_fundamentals.jl with same filename
3. Run text_to_music.py if fails, otherwise just execute.py
4. Check new piano_score.pdf


## Hierarchical knowledge


# Run the clustering algorithm written in Julia on the saved CSV's for the c_<file>.csv 
# and test the hierarchical knowledge representation in knowledge_heararchy.ipynb

# Tutorial for installing julia for python: https://www.peterbaumgartner.com/blog/incorporating-julia-into-python-programs/

# first add path to julia by: 'export PATH=$PATH:/Users/<USERNAME>/.local/bin' if code doesn't work