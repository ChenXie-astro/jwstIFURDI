from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# setup
setup(
    name='jwstIFURDI',
    version='1.0',
    description='Reference-star differential imaging on JWST/NIRSpec IFU',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ChenXie-astro/jwstIFURDI',
    author='Chen Xie',
    author_email='cx@jhu.edu',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research ',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
    ],
    keywords='jwst NIRSpec ifu rdi reduction disk pipeline',
    packages=['jwstIFURDI'],
    install_requires=[
        'numpy', 'scipy', 'astropy', 'pandas', 'matplotlib', 'scikit-image', 'debrisdiskfm', 'emcee', 'corner', 'diskmap'
    ],
    include_package_data=True,
    # package_data={
    #     'sphere': ['data/*.txt', 'data/*.dat', 'data/*.fits',
    #                'instruments/*.ini', 'instruments/*.dat'],
    # },
    zip_safe=False
)