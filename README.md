# xgapfilling
Gap filling using direct sampling and extremes. Explore fracture density reconstruction with Jupyter notebook examples and synthetic well log data. This repository is associated with paper under review:

**Fracture density reconstruction using direct sampling multiple-point statistics and extreme value theory.**

Ana Paula Burgoa Tanaka <sup>1,2\*</sup>, Philippe Renard<sup>1</sup>, Julien Straubhaar<sup>1</sup>

<sup>1 </sup> Centre for Hydrogeology and Geothermics, University of Neuchâtel, Neuchâtel, Switzerland  
<sup>2 </sup> Santos Basin Unit, Petrobras, Santos, São Paulo, Brazil   

\* corresponding author: ana.burgoa@unine.ch

## Journal reference
A. P. B. Tanaka, P. Renard, J. Straubhaar (2024). Fracture Density Reconstruction Using Direct Sampling Multiple-Point Statistics and Extreme Value Theory. In revision in *Applied Computing and Geosciences* - Ref.: ACAGS-D-23-00073 - March 2024.

## Data description
Due to restrictions on sharing the original data used in the reference, we generate synthetic data for the examples provided in this repository. The synthetic logs analysis do not add content to the manuscript. It is available to disseminate the application of the proposed methodology and to easily run the code. People are also encouraged to try with their own data set.

### Files

In the "data" folder, you will find:

- Synthetic well log fracture density data in gslib format
  - Filename: `syn_well_p10.gslib`

- Synthetic well log fracture density data with gap in gslib format
  - Filename: `syn_well_p10_gap.gslib`

- Synthetic well log enriched fracture density data with gap in gslib format
  - Filename: `syn_naive.gslib`
 
- Synthetic well log enriched fracture density data with gap in gslib format
  - Filename: `syn_lift.gslib`

## Dependencies

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following packages.

```bash
pip install geone matplotlib numpy scipy jupyter-notebook
```
