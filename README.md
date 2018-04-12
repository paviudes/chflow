# chflow -- Quantum error correction for realistic noise

[![DOI](https://zenodo.org/badge/125156309.svg)](https://zenodo.org/badge/latestdoi/125156309)

This package is intended to be used for studying the response of different quantum error correction schemes on various types of quantum noise processes. In order to gauge the response of a scheme, we perform numerical simulations of a quantum (stabilizer) error correcting scheme over a noise model and study different measures of noise strength for a qubit.

## Example
As an example, the following plot produced by `chflow` describes the [fidelity](https://github.com/paviudes/chflow/wiki/Measures-of-noise-strength) of a logical qubit [error corrected](https://github.com/paviudes/chflow/wiki/Running-Simulations#running-simulations) for [coherent errors](https://github.com/paviudes/chflow/wiki/Quantum-channels#predefined-channels) \(rotations about Z-axis\) as a function of the angle of the over rotation error.
![rotz](https://github.com/paviudes/chflow/blob/master/docs/rotz.jpg)

The following command set is required to generate the above plot.

```python
Pavithrans-MacBook-Pro:chflow pavithran$ ./chflow.sh
>> sbload 11_04_2018_13_52_37
Simulation data is available for 0% of the channels.
Preparing physical channels... 26 (100%) done. 
>> sbprint
Time stamp: 11_04_2018_13_52_37
 Physical channel
Parameters                     Values                        
Channel                        rtz                           
Noise range                    [ 0.05  0.06  0.07  0.08  0.09  0.1  0.11  0.12  0.13  0.14  0.15  0.16  0.17  0.18  0.19  0.2  0.21  0.22  0.23  0.24  0.25  0.26  0.27  0.28  0.29
  0.3].
Scales of noise rates          [ 1.0]                        
Number of Samples              1                             
Metrics
Logical metrics                fidelity                      
Error correction
QECC                           Steane X Steane               
[[N, K, D]]                    [[49, 1, 9]]                  
Levels of concatenation        2                             
ECC frame                      P                             
Decoder                        0                             
Syndrome samples at level 2    10000                         
Type of syndrome sampling      N                             
 >> ecc
Simulation data is available for 0% of the channels.
Please wait ...
100% done, approximately 0 seconds remaining ...   
done, in 870 seconds.
>> metrics fidelity 
>> collect
Simulation data is available for 100% of the channels.
>> lplot 0 fidelity
>> 
```

For any stabilizer error correction scheme, `chflow` tracks the _flow of a quantum channel_ -- evolution from a physical noise process to the _effective quantum channel_ that affects the underlying logical information encompassing quantum error correction steps.

More about quantum error correction with various noise processes can be learnt from [arXiv:1711.04736](https://arxiv.org/abs/1711.04736) and [Phys. Rev. A 95, 042332](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.95.042332).

In addition to its main purpose, this package can also be used to perform simple but useful operations on quantum error correcting codes as well as quantum channels. For instance, derivation of the [canonical basis](https://github.com/paviudes/chflow/wiki/Quantum-error-correction#complete-description-of-a-stabilizer-code) \(Stabilizers, Logicals and Pure errors\) for an error correcting code, converting between various [representations of a quantum channel](https://github.com/paviudes/chflow/wiki/Quantum-channels#representations-for-quantum-channels), computing [Pauli approximations](https://github.com/paviudes/chflow/wiki/Quantum-channels#approximations-to-a-pauli-channel) of a quantum channel and so on.

## Downloading and installing
The latest version of `chflow` can be obtained by either [cloning this github repository](https://help.github.com/articles/cloning-a-repository/) or directly downloading the source zip file by following the [![CloneDownload](https://img.shields.io/badge/-Clone%20or%20download-brightgreen.svg)](https://github.com/paviudes/chflow) link at the top of this page.

The following dependencies, along with their recommended versions
[![Python](https://img.shields.io/badge/Python-2.7-Green.svg)](https://www.python.org/downloads/)
[![Numpy](https://img.shields.io/badge/Numpy-1.1.0-Red.svg)](https://www.scipy.org/install.html)
[![PICOS](https://img.shields.io/badge/PICOS-1.1.2-Green.svg)](http://picos.zib.de/intro.html#installation)
[![CVXOPT](https://img.shields.io/badge/CVXOPT-1.1.9-Green.svg)](http://cvxopt.org/install/index.html)
[![Cython](https://img.shields.io/badge/Cython-0.25.2-Red.svg)](https://docs.anaconda.com/anaconda/install/)
are desirable for the smooth compiling and execution of `chflow`. Refer to the [Wiki](https://github.com/paviudes/chflow/wiki) for further instructions on running `chflow`.

## Contributing to `chflow`

There are no restrictions on contributing to `chflow`. While updating the GitHub repository, care has to be taken to avoid uploading large (input and output) data files that are generated from Quantum error correction simulations. To make this easier, there is a `clean` command available in the `chflow` interface. Please run `clean git` in `chflow`, which moves all the unwated files (for publishing a release) into a folder called `.gitignore` and delete this folder using `rm -r .gitignore` before pushing on to the GitHub repository.

## Citing `chflow`
```text
P. Iyer and D. Poulin. chflow: quantum error correction for realistic noise. https://github.com/paviudes/chflow, DOI: 10.5281/zenodo.1216202, April 2018.
```
```text
@misc{chflow,
  author       = {Pavithran Iyer and David Poulin},
  title        = {chflow: Quantum error correction for realistic noise},
  month        = {April},
  year         = {2018},
  doi          = {10.5281/zenodo.1216202},
  url          = {https://github.com/paviudes/chflow}
}
```
