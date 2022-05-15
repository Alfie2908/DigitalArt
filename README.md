Evolutionary Algorithms for Recreating Art
==========================================

In the repository there are five algorithms for evolving Art, the algorithms have 100
semi transparent polygons to draw in order to replicate a given image.

All algorithms are set to run until 0.95 fitness is reached or it exceeds 6000 
generations on Charles Darwin, but this can be changed in the source code.

1. reference.py - Basic Evolution
2. representation _variant.py- Similar to basic but uses ellipses instead
3. mutation_variant.py - Update on crossover and mutation functions
4. selection_variant_ranked.py - Uses ranked selection
5. selection_variant_tournament.py - Uses tournament selection

To run the variants enter this command in the terminal:

~~~python
 python3 (name of algorithm).py config.ini (config header)
~~~

Where (name of algorithm) is one of the above files, and (config header) is one of 
reference, representation, mutation, selection_ranked, selection_tournament.