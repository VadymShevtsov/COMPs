# COMPs

This work presents an addition to the family of modules for IGA analysis in the form of composite material definition. It also includes a CDM model to demonstrate how complex material models can be further implemented. The formulation and explanation of the code structure can be found in:

```
@article{Zhao2022,
title = "An open-source framework for coupling non-matching isogeometric shells with application to aerospace structures",
journal = "Computers \& Mathematics with Applications",
volume = "111",
pages = "109--123",
year = "2022",
issn = "0898-1221",
doi = "https://doi.org/10.1016/j.camwa.2022.02.007",
author = "H. Zhao and X. Liu and A. H. Fletcher and R. Xiang and J. T. Hwang and D. Kamensky"
}
```
The examples include: plate under uniform edge pressure, plate under uniform out-of-plane pressure, and plate with a hole. Additional guides for installation, visualization, and mortar analysis are present in the form of .txt files.

## Dependencies 
1. Python library for FE analysis [FEniCS](https://fenicsproject.org/).

3. An addition to FEniCS to perform IGA analysis [tIGAr](https://github.com/david-kamensky/tIGAr).

4. Kirchhoff--Love shell formulation for tIGAr is given by [ShNAPr](https://github.com/david-kamensky/ShNAPr).

5. Multipach analysis requires [PENGoLINS](https://github.com/hanzhao2020/PENGoLINS).

6. IGA geometry is created using [IGAkit](https://github.com/dalcinl/igakit).




