## Sanity check evaluation

This section contains comparison scores, obtained on a small subset of the [NDIris3D dataset](https://cvrl.nd.edu/projects/data/#ndiris3d), for C++ versions of both CRYPTS and HDBIF methods submitted to IREX X. These scores may be useful in comparing whether compiled IREX X submissions of CRYPTS and HDBIF generate the same scores on your end as on our end.

## Steps

1. Request the [NDIris3D dataset](https://cvrl.nd.edu/projects/data/#ndiris3d).
2. Image pairs for which the scores were calculated are defined in `./CRYPTS/crypts_pairs.txt` and `./HDBIF/hdbif_pairs.txt`. These files also contain the scores.
3. Run your method to obtain comparison scores on the same pairs.
4. To generate ROC curve and histogram, run:

```
python generate_plots.py <path-to-resulting-pairs.txt> <path-where-roc-and-histogram-are-to-be-saved>
``` 
