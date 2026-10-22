Goal: To get the fields of all predicted years (here, 2021, 2023 and 2025) with less noise and clear geometry:
Input: binary raster - 1 lavender, 0 background

1. Clean the raster before vectorisation 
2. Convert the .tif raster into vector 
2. Remove background
3. Remove very thin objects (stripes: Minimum estimated width or elongation to width ratio)
4. Remove very small fields (say, less than 0.5 ha)
5. Simplify geometry to represent only 2 nodes to an edge
6. Restore the shape of the fields of irregular (e.g., triangle fields) or not fully shaped (concave fields with a lot of vertices in a concave part) and using the initial image
7. Add the statistics to the vector file: Area