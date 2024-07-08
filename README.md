# jwstIFURDI
Reference-star differential imaging on JWST/NIRSpec IFU

Information
-----------

jwstIFURDI was designed to perform high-contrast imaging for the [JWST/NIRSpec IFU](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph#gsc.tab=0) observations to extract disk reflectance spectrum.  

Point spread function (PSF) subtraction is essential to detect faint planets and circumstellar disks because the NIRSpec IFU does not have a coronagraph to suppress the light from the host star. 
We performed PSF subtraction from every spectral channel using reference-star differential imaging (RDI) to remove the stellar PSF in the spatial direction.
To perform RDI, the NIRSpec IFU observations require two targets, a science star and a reference star to provide the empirical PSF (i.e., PSF calibrator).

If you find a bug or want to suggest improvements, please [create a ticket](https://github.com/ChenXie-astro/jwstIFURDI/issues).


Installation
------------
```
% pip install git+https://github.com/ChenXie-astro/jwstIFURDI.git#egg=jwstIFURDI
```

Requirements
------------
  - numpy
  - scipy
  - matplotlib
  - scikit-image
  - [astropy](https://www.astropy.org)
  - [diskmap](https://diskmap.readthedocs.io/en/latest/)
  - [debrisdiskfm](https://github.com/seawander/DebrisDiskFM)
  - [emcee](https://emcee.readthedocs.io/en/stable/)
  - [corner](https://corner.readthedocs.io/en/latest/)

Tested: Python 3.7.7, numpy 1.21.6, scipy 1.7.3, matplotlib 3.3.0, scikit-image 0.16.2, astropy 3.2.3, diskmap 0.1.5, debrisdiskfm 0.1.0, emcee 3.0rc1, corner 2.1.0.


Run MCMC disk modeling
----------------------
To estimate the flux loss caused by RDI, we need to perform the negative injection. \
The following code performs the negative disk injection using the MCMC analysis to estimate the disk parameters. \
In addition to the science and reference data cubes, we also need to provide the empirical IFU PSF and the uncertainty map/cube. \

To perform MCMC disk modeling
```
% python MCMC_disk_modeling.py
```
It will create four files: (1) best-fit disk parameters obtained with MCMC; (2) reduced chi-square and corresponding disk flux scaling factor; (3) residual images of the disk-subtracted data cube; and (4) best-fit disk model (PSF convolved). All four files are wavelength-dependent. The uncertainty map/cube can be estimated by applying the standard deviation per spatial pixel in the spectral direction on the residual cube of the disk-subtracted data cube (Step 5)

**Note**: it is highly recommended to run the MCMC code on a computer cluster instead of a personal laptop. 

MCMC disk modeling provides two necessary files (the best-fit disk model and the residual cube of the disk-subtracted data cube) to perform the throughput correction and uncertainty estimation for the extracted disk spectrum. 

Run RDI and extract the target spectrum 
---------------------------------------
To perform RDI and extract the target spectrum
```
% python run_jwstIFURDI.py
```
Post-processing parameters can be modified in the Python script. The above process can be divided into 6 steps.

**Step 1**: image alignment\
Info: centering both science and reference cubes to the image center. The star position was determined by using the diffraction spikes and [centerRadon](https://github.com/seawander/centerRadon).\
Output: aligned science and reference cubes

**Step 2**: RDI subtraction\
RDI: residual_image  = sicence_image - scaling_factor * refernce_image\
Info: Step 2 can be performed right after step 1.\
Input: aligned science and reference cubes, a spike mask, and an annular mask for fitting the scaling factor in RDI\
Output: residual disk cube after RDI

**Step 3**: calculate the PSF convolution effect\
Info: IFU PSF is wavelength dependent and a fixed aperture was used in extracting the disk spectrum. Thus, the PSF convolution effect needs to be corrected.\
Input: disk model parameters and masks of disk spectral extracting regions\
Output: The PSF convolution correction factor as a function of wavelength\
Step 3 prepares the correction file for Step 6 

**Step 4**: create a r^0.5 correction map to correct the illumination effect\
This step uses [diskmap](https://diskmap.readthedocs.io/en/latest/) to calculate the stellocentric distances, thus the r^0.5 correction map.\
Input: disk parameters
Output: radius map and r^0.5 correction map. \
Step 4 prepares the correction file for Step 6 

**Step 5**: calculate the uncertainty cube for MCMC disk modeling \
The uncertainty map/cube can be estimated by applying the standard deviation per spatial pixel in the spectral direction on the residual cube of the disk-subtracted data cube\
Input: MCMC results \
Output: uncertainty cube \
Step 5 prepares the uncertainty cube for MCMC disk modeling 

**Step 6**: 
Extracting the disk reflectance spectrum with four flux corrections (see Xie et al., **in prep** for details)\
Input: residual disk cube (from Step 2), residual disk-free cube (from MCMC disk modeling), best-fit disk model (from MCMC disk modeling), stellar photosphere model, PSF convolution correction cube (from Step 3), the illumination correction map (from Step 4), masks of disk spectral extraction regions.
Output: disk reflectance spectrum, RDI throughout 



Credits
-------
If you use DDRM in your research, please cite

Xie et al. **in prep.**



In addition:
If you aligned the cube using **jwstIFURDI.centering**, please also cite
```
@INPROCEEDINGS{2017SPIE10400E..21R,
       author = {{Ren}, Bin and {Pueyo}, Laurent and {Perrin}, Marshall D. and {Debes}, John H. and {Choquet}, {\'E}lodie},
        title = "{Post-processing of the HST STIS coronagraphic observations}",
     keywords = {Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Earth and Planetary Astrophysics},
    booktitle = {Society of Photo-Optical Instrumentation Engineers (SPIE) Conference Series},
         year = 2017,
       editor = {{Shaklan}, Stuart},
       series = {Society of Photo-Optical Instrumentation Engineers (SPIE) Conference Series},
       volume = {10400},
        month = sep,
          eid = {1040021},
        pages = {1040021},
          doi = {10.1117/12.2274163},
archivePrefix = {arXiv},
       eprint = {1709.10125},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2017SPIE10400E..21R},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

If you perform the disk modeling using **jwstIFURDI.MMB_anadisk_model**, please also cite
```
@ARTICLE{2015ApJ...811...18M,
       author = {{Millar-Blanchaer}, Maxwell A. and {Graham}, James R. and {Pueyo}, Laurent and {Kalas}, Paul and {Dawson}, Rebekah I. and {Wang}, Jason and {Perrin}, Marshall D. and {moon}, Dae-Sik and {Macintosh}, Bruce and {Ammons}, S. Mark and {Barman}, Travis and {Cardwell}, Andrew and {Chen}, Christine H. and {Chiang}, Eugene and {Chilcote}, Jeffrey and {Cotten}, Tara and {De Rosa}, Robert J. and {Draper}, Zachary H. and {Dunn}, Jennifer and {Duch{\^e}ne}, Gaspard and {Esposito}, Thomas M. and {Fitzgerald}, Michael P. and {Follette}, Katherine B. and {Goodsell}, Stephen J. and {Greenbaum}, Alexandra Z. and {Hartung}, Markus and {Hibon}, Pascale and {Hinkley}, Sasha and {Ingraham}, Patrick and {Jensen-Clem}, Rebecca and {Konopacky}, Quinn and {Larkin}, James E. and {Long}, Douglas and {Maire}, J{\'e}r{\^o}me and {Marchis}, Franck and {Marley}, Mark S. and {Marois}, Christian and {Morzinski}, Katie M. and {Nielsen}, Eric L. and {Palmer}, David W. and {Oppenheimer}, Rebecca and {Poyneer}, Lisa and {Rajan}, Abhijith and {Rantakyr{\"o}}, Fredrik T. and {Ruffio}, Jean-Baptiste and {Sadakuni}, Naru and {Saddlemyer}, Leslie and {Schneider}, Adam C. and {Sivaramakrishnan}, Anand and {Soummer}, Remi and {Thomas}, Sandrine and {Vasisht}, Gautam and {Vega}, David and {Wallace}, J. Kent and {Ward-Duong}, Kimberly and {Wiktorowicz}, Sloane J. and {Wolff}, Schuyler G.},
        title = "{Beta Pictoris' Inner Disk in Polarized Light and New Orbital Parameters for Beta Pictoris b}",
      journal = {\apj},
     keywords = {astrometry, planet{\textendash}disk interactions, planets and satellites: individual: {\ensuremath{\beta}} Pic b, techniques: polarimetric, Astrophysics - Earth and Planetary Astrophysics},
         year = 2015,
        month = sep,
       volume = {811},
       number = {1},
          eid = {18},
        pages = {18},
          doi = {10.1088/0004-637X/811/1/18},
archivePrefix = {arXiv},
       eprint = {1508.04787},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2015ApJ...811...18M},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
and
```
@misc{debrisdiskFM,
  author       = { {Ren}, Bin and {Perrin}, Marshall },
  title        = {DebrisDiskFM, v1.0, Zenodo,
doi: \href{https://zenodo.org/badge/latestdoi/141328805}{10.5281/zenodo.2398963}. },
  version = {1.0},
  publisher = {Zenodo},
  month        = Dec,
  year         = 2018,
  doi          = {10.5281/zenodo.2398963},
  url          = {https://doi.org/10.5281/zenodo.2398963}
}
```


If you use **emcee** in performing the MCMC analysis, please also cite
```
@ARTICLE{2013PASP..125..306F,
       author = {{Foreman-Mackey}, Daniel and {Hogg}, David W. and {Lang}, Dustin and {Goodman}, Jonathan},
        title = "{emcee: The MCMC Hammer}",
      journal = {\pasp},
     keywords = {Astrophysics - Instrumentation and Methods for Astrophysics, Physics - Computational Physics, Statistics - Computation},
         year = 2013,
        month = mar,
       volume = {125},
       number = {925},
        pages = {306},
          doi = {10.1086/670067},
archivePrefix = {arXiv},
       eprint = {1202.3665},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2013PASP..125..306F},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

If you use **corner** to make the corner plot, please also cite
```
  @article{corner,
      doi = {10.21105/joss.00024},
      url = {https://doi.org/10.21105/joss.00024},
      year  = {2016},
      month = {jun},
      publisher = {The Open Journal},
      volume = {1},
      number = {2},
      pages = {24},
      author = {Daniel Foreman-Mackey},
      title = {corner.py: Scatterplot matrices in Python},
      journal = {The Journal of Open Source Software}
    }
```

If you use **diskmap** to calculate the stellocentric distance, please also cite
```
@ARTICLE{2016A&A...596A..70S,
       author = {{Stolker}, T. and {Dominik}, C. and {Min}, M. and {Garufi}, A. and {Mulders}, G.~D. and {Avenhaus}, H.},
        title = "{Scattered light mapping of protoplanetary disks}",
      journal = {\aap},
     keywords = {protoplanetary disks, scattering, polarization, stars: individual: HD 100546, methods: numerical, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Solar and Stellar Astrophysics},
         year = 2016,
        month = dec,
       volume = {596},
          eid = {A70},
        pages = {A70},
          doi = {10.1051/0004-6361/201629098},
archivePrefix = {arXiv},
       eprint = {1609.09505},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2016A&A...596A..70S},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```