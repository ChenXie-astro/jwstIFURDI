# jwstIFURDI
Reference-star differential imaging on JWST/NIRSpec IFU

Information
-----------

jwstIFURDI was designed to perform high-contrast imaging for J[JWST/NIRSpec IFU](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph#gsc.tab=0) observations to extract disk reflectance spectrum.  

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
  - [diskmap]()
  - [debrisdiskfm]()
  - [emcee](https://emcee.readthedocs.io/en/stable/)
  - [corner](https://corner.readthedocs.io/en/latest/)

Tested: Python 3.7.7, numpy 1.21.6, scipy 1.7.3, matplotlib 3.3.0, scikit-image 0.16.2, astropy 3.2.3, diskmap 0.1.5, debrisdiskfm 0.1.0, emcee 3.0rc1, corner 2.1.0.


Run MCMC disk modeling
----------------------
Info:

```
% python MCMC_disk_modeling.py
```

Output

Note

Run RDI and extract the target spectrum 
-----------------------------------
Info 

Inputs

```
% python run_jwstIFURDI.py
```
Post-processing parameters can be modified in the Python script. The aboved Python script can be divded into 6 steps.

Details:

Step 1

Outputs:

Step 2

Outputs:

Step 3

Outputs:

Step 4

Outputs:


Step 5

Outputs:

Step 6

Outputs:


Credits
-------
If you use DDRM in your research, please cite

Xie et al. **in prep.**



In addition:
If you aligned the cube using jwstIFURDI.centering, please also cite
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

If you perform the disk modeling using jwstIFURDI.MMB_anadisk_model, please also cite
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

If you use emcee in performing the MCMC analysis, please also cite
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

If you use corner to make the corner plot, please also cite
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
