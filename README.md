## LUE-patterns
Regression on LUE eigenvalues

We implement [Anna Maltsev's](https://webspace.maths.qmul.ac.uk/a.maltsev/) idea of fitting a linear model (doing linear regression) on the Laguerre Unitary Ensemble eigenvalues for a fixed parameter $\alpha$ ($=4$ in this case). More details are at the beginning of each notebook. 

**Some remarks:**

- the data contains three sets of files, for number of eigenvalues $N=1000, 5000, 10000$. For each $N$, due to Github restrictions on file size, there are 4 files which need to be concatenated. Under Linux or MacOS you can simply run the following three separate commands:

```
cat LUE_N_1000_alpha_4.00_xa*.txt > LUE_N_1000_alpha_4.00.txt
cat LUE_N_5000_alpha_4.00_xa*.txt > LUE_N_5000_alpha_4.00.txt
cat LUE_N_10000_alpha_4.00_xa*.txt > LUE_N_10000_alpha_4.00.txt
```

- we use both Julia and Python
- if the difference between two files is ```julia``` vs ```python``` in the file name, they do the same thing, but in slightly different ways
- there are Markdown files (ending in ```.md``` in the ```notebooks``` folder) which, when clicked, give a pretty neat webpage of each individual notebook
- this repository is constantly changing/being refined as more tests are being performed
