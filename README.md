# FlexKnotFRB

These are the codes used for our paper "What it takes to measure reionization with fast radio bursts". If you would like to use any of these methods please feel free to contact me for more details and explanations, my email address can be found [here](https://www.ast.cam.ac.uk/people/Stefan.Heimersheim) or you can just open an issue on here GitHub :)

## Generation of the synthetic data set
The codes `A_create_perfect_data.py` and `B_generate_synthetic_observations.py`
are responsible for generating the synthetic data sets. The former computes
the mean (homogeneous) expectation, and the latter adds errors and uncertainties.
The star formation rate is obtained from [Behroozi et al. 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.3143B/abstract) (the file is `umachine-dr1/data/csfrs/csfrs.dat` from [UniverseMachine DR1](https://www.peterbehroozi.com/data.html), [direct link](https://slac.stanford.edu/~behroozi/UniverseMachine/umachine-dr1.tar.gz)), and
the ionization history from [Kulkarni et al. 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.485L..24K/exportcitation),
both files are included in the repository.
```bash
mkdir -p data/ # in case you did not have the folder
python3 A_create_perfect_data.py
python3 B_generate_synthetic_observations.py
```
This plot (also generated by the 2nd script) shows the generated synthetic observations, compared to the
true locations and DM expectation values of the sources. Note that all 3 z>10 FRBs (in the 100 FRB data set) happened to scatter
downwards (depends on the random seed, expect something like this in 1/4 cases, we just stuck with the first
data set generated) so if you use this to e.g. infer optical depth you might find the likelihood peaks at
slightly lower optical depth than the fiducial 0.057, but this is perfectly fine and within the uncertainties.
![Figure_1](https://user-images.githubusercontent.com/40799217/127821389-b61b0404-1833-48e9-8b60-baf9e4e51603.png)


## Running the Nested Sampling analysis
To run the Nested Sampling chains use `D_run.py` (`C_likelihood.py` is
just the likelihood implementation). You can use it like this:
```bash
python3 D_run.py Mockdata_phys_sfr_z10%_norm100_obs1.npy planckpriors monotonous flexknot2 eprior varyzend hostcalx12

#  Mockdata_phys_sfr_z10%_norm100_obs1.npy is the file with synthetic data being loaded
#  planckpriors indicates to use Planck constraints for Omega_m and Omega_b_H0
#  flexknot2 indicates using the FlexKnot algorhtm, and the number 2 gives the
#            "effective" number of knots, or the degrees of freedom divided by 2.
#            So "1" refers to moving start+end point, "2" adds one fully movable knot.
#  The 'monotonous' keyword enables enforcing a monotonous ionization history.
#  eprior indicates to use the common z*exp(-z) formulation for the prior on the FRB source distribution,
#         instead of the true prior distribution which might be unknown.
#  varyzend indicates moving the start and endpoints, otherwise they are fixed to redshifts 5 and 30, respectively.
#  I use the argument including 'host' (e.g. hostcalx12) to decide the output filename.
```
The example python line will take the 100 FRB data set, and run a
Nested Samplung run with 3 FlexKnots (Start point, end point and one
fully movable point) and vary Omega_m and Omega_b_H0 within priors
approximately (i.e. assuming no correlation) corresponding to Planck limits.

## Analysing the Nested Samplung runs and generating posterior constraints
All Figures and Numbers from the paper can be generated using the analysis script.
```bash
python3 E_analysis.py
```
You can skip generating and fitting new priors using options in the file:
```python
# Whether or not to generate new samples from the FlexKnot prior, and make new fits for cancelling the prior effect
regeneratePriors = False #new prior samples
regenerateTau = False #recompute tau for prior samples
redoFits = False #redo prior distribution fits
```
Running the script should take about 5-10 minutes, assuming you are using the data from our [zenodo record](https://zenodo.org/record/5504050).

That [zenodo record](https://zenodo.org/record/5504050) contains the NestedSampling chains as well as the pre-computed plotting data and fits (`cache` folder).

Note: The numbers here differ slightly from the initial arXiv version (will be updated with final version) mainly due to 2 reasons:
1. We now use the conventional definition of confidence intervals (or, more precisely, credibility intervals) defined with iso-probability boundaries, and we quote the maximum posterior probability points instead of mean values.
2. We use Omega_b *times* H_0 as variable, not Omega_b *over* H_0. This does not change anything significantly but makes the prior on the DM-"prefactor" narrower (as Omega_b times H0 is constraint better than Omega_b over H0). Omega_b *times* H_0 is the correct dependence of the DM amplitude (our previous formulation did not take into account that the cosmological critical density depends on H0^2).
