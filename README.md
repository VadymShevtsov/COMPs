# COMPs

This work presents an addition to the family of modules for IGA analysis in the form of composite material definition. It also includes a CDM model to demonstrate how complex material models can be further implemented. The formulation and explanation of the code structure can be found in:

```
@mastersthesis{Shevtsov2024,
    author = {Shevtsov V.},
    institution = {Master's thesis},
    pages = 76,
    school = {University of Calgary},
    title = {Open-source isogeometric composite module with continuum degradation model},
    note = {Available at \url{https://hdl.handle.net/1880/119758}},
    year = 2024
}
```
Examples include: plate under uniform edge pressure, plate under uniform out-of-plane pressure, and plate with a hole. Additional guides for installation, visualization, and mortar analysis are present in the form of .txt files.

## Dependencies 
1. Python library for FE analysis [FEniCS](https://fenicsproject.org/).

3. An addition to FEniCS to perform IGA analysis [tIGAr](https://github.com/david-kamensky/tIGAr).

4. Kirchhoff--Love shell formulation for tIGAr is given by [ShNAPr](https://github.com/david-kamensky/ShNAPr). Importantly, the covariantRank2TensorToCartesian2D(...) function should be modified when working with complex geometries; for additional guidance, check the reference.

5. Multipach analysis requires [PENGoLINS](https://github.com/hanzhao2020/PENGoLINS).

6. IGA geometry is created using [IGAkit](https://github.com/dalcinl/igakit).




