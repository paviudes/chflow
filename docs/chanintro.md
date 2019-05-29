Arbitray precision control of quantum systems is quite a difficult task. A quantum system constantly interacts with its environment. While assuming that the system and its environment together form a closed quantum system, thier dynamics can be modelled by unitary operators. Our ignorance of the state of the enviornment is mathematically equivalent to performing a partial trace over the subspace of the closed system that belongs to the environment. Consequently, the result of the unitary dynamics and partial trace corresponds to a [_completely positive trace preserving (CPTP) map_](https://www.sciencedirect.com/science/article/pii/0024379575900750). The type of noise processes studied in `chflow` include all CPTP maps at the level of a physical qubit. In the [table](https://github.com/paviudes/chflow/wiki/Quantum-channels#predefined-channels) below, we have listed some of the commonly known noise processes and how they can be recalled within `chflow`.

While studying errors on a quantum state with many physical qubits, eg. encoded states, we will assume that each of the constituent physical qubits are affected by the same noise process. This type of noise is termed as _independent and identically distributed (i.i.d)_ a term borrowed from classical terminology.

## Defining a quantum channel
### Predefined channels
Definitions for several noise prcesses have been preloaded in `chflow` and they can be recalled using their corresponding names. The table below provides these names. The most popular representation for quantum noise processes is called the Krauss representation. The document [cdefs.pdf](https://github.com/paviudes/chflow/blob/master/docs/cdefs.pdf) contains the Krauss representations for the predefined noise processes in the table below.

| Name                    		| Description                                               	| Number of parameters 	|
|-------------------------------|---------------------------------------------------------------|----------------------	|
| `ad`                      	| Amplitude dampling channel                                	| 1                    	|
| `bp`                      	| Bit flip channel                                          	| 1                    	|
| `pd`                      	| Dephasing channel                                         	| 2                    	|
| `bpf`                     	| Bit-Phase flip channel                                    	| 2                    	|
| `gd`                      	| Generalized damping channel                               	| 3                    	|
| `gdt`                     	| Generalized time dependent damping channel                	| 1                    	|
| `gdtx`                    	| Generalized damping channel with time scales              	| 1                    	|
| `dp`                      	| Depolarizing channel                                      	| 1                    	|
| `pauli`                   	| Generic Pauli channel                                     	| 3                    	|
| `rtx`,`rty`,`rtz`             | Fixed angle rotations about X, Y and Z axis respectively  	| 1                    	|
| `rtxpert`,`rtypert`,`rtzpert` | Random angle rotations about X, Y and Z axis respectively 	| 1                    	|
| `rtnp`                    	| Rotations about a non-Pauli axis                          	| 3                    	|
| `strtz`                   	| Stochastic rotation about Z-axis                          	| 2                    	|
| `pl`                      	| Photon loss channel                                       	| 2                    	|
| `rand`                    	| Random CPTP map                                           	| 2                    	|

The `chan` command can be used to recall any of the predefined channels in the following way: `chan <name> <parameter values separated by commas>`. A few examples are shown in the figure below.

```python
Pavithrans-MacBook-Pro:chflow pavithran$ ./chflow.sh 
>> chan rand 0.2,1
Note: the current channel is in the "krauss" representation.
>> chprint
Krauss representation
E_1
[[ 0.862-0.284j  0.120+0.049j]
 [-0.132+0.13j   0.855-0.064j]]
E_2
[[ 0.014-0.026j  0.121-0.084j]
 [ 0.075-0.027j -0.183+0.052j]]
E_3
[[-0.231+0.055j -0.195+0.106j]
 [-0.060+0.002j  0.170-0.153j]]
E_4
[[ 0.032+0.043j -0.193+0.219j]
 [-0.239+0.125j -0.025-0.05j ]]
xxxxxx
>> 
```
***

### User defined channels

In addition to the predefined set of channels, it is also possible to define new quantum channels by specifying it in one of the representations. There are two formats to define a quantum channel.
1. An explicit specification of the channel representation in numeric form. The associated description must be stored in a file (numpy formatted file: `.npy` or a text file) along with the representation being used. When unspecified, the default representation for quantum channels is the Pauli Liouville representation. The definition can then be recalled in `chflow` as: `chan <file name>` where file name must include its extension.
2. A symbolic form of the channel representation, an array containing numbers as well as variables or python interpretable expressions of variables. In this case, the list of variables must be provided at first, in the same order as the list of parameters to be input while recalling the channel definition using `chan` (as in the figure above). Here again, the channel description must be contained in a text file, first specifying the variable symbols with the `vars` keyword, as: `vars <v1> <v2> <v3> ...` where `<v1>`, `<v2>`, ... must be replaced by variable symbols. The following lines should contain the array elements, each row spanning a new line and each column entry separated by a space. An example of a text file used to define a quantum channel that performs the Clifford operations S and H with probabilities p and q respectively (see Section 2 of channeldefs.pdf for details), in its process matrix representation is shown below.
```text
# user defined quantum channel
# applies S and H with probabilities p and q respectively
vars p q
1 0 0 0
0 1-p-q p q
0 -p 1-p-2*q 0
0 q 0 1-q
```
Finally, the above quantum channel can be recalled in `chflow` by: `chan <file name with relative path> <value for p>,<value for q>`.

# Representations for quantum channels
We will consider quantum channels that can be described by completely positive trace preserving (CPTP) maps. A single qubit CPTP map can be represented in one of the following representations.
1. Krauss representation (`krauss`)
2. Choi matrix (`choi`)
3. Pauli Liouville matrix (`process`)
4. Chi Matrix (`chi`)
5. Stinespring dilation (`stine`)

Refer to the document [creps.pdf](https://github.com/paviudes/chflow/blob/master/docs/creps.pdf) for definitions of the above representations. In chflow, when a channel is defined using the `chan` command, it is stored in the Krauss representation. The storage representation can be altered using the `chrep` command. In order to alter the representation from `<oldrep>` to `<newrep>`, use the command

`chrep <oldrep> <newrep>`.

```python
Pavithrans-MacBook-Pro:chflow pavithran$ ./chflow.sh 
>> chan rand 0.2,1
Note: the current channel is in the "krauss" representation.
>> man chrep
	"chrep"
	Description: Convert from its current representation to another form.
	Usage
	chrep s1(string)
	where s1 must be one of "krauss", "choi", "chi", "process", "stine".
xxxxxx
>> chrep choi
>> chprint
Choi representation
[[ 0.398+0.j     0.130-0.049j -0.172-0.006j  0.342-0.108j]
 [ 0.130+0.049j  0.085+0.j    -0.059-0.025j  0.156+0.004j]
 [-0.172+0.006j -0.059+0.025j  0.102+0.j    -0.130+0.049j]
 [ 0.342+0.108j  0.156-0.004j -0.130-0.049j  0.415+0.j   ]]
xxxxxx
>> chrep process
>> chprint
Pauli Liouville representation
[[ 1.0 -0.0327  0.00258 -0.0337]
 [-1.67e-16  0.565  0.266  0.521]
 [ 5.55e-17 -0.165  0.803 -0.194]
 [ 5.55e-17 -0.655  0.0206  0.627]]
xxxxxx
>> chrep chi
>> chprint
Chi representation
[[ 0.749+0.0j -0.00818+0.0537j  0.000645+0.294j -0.00842-0.108j]
 [-0.00818-0.0537j  0.034+0.0j  0.0253-0.00842j -0.0335-0.000645j]
 [ 0.000645-0.294j  0.0253+0.00842j  0.153+0.0j -0.0435-0.00818j]
 [-0.00842+0.108j -0.0335+0.000645j -0.0435+0.00818j  0.0647+0.0j]]
xxxxxx
>> 
```
Check `man chrep` for details.

# Approximations to a Pauli channel

In several situations, it is desirable to have noise processes as CPTP maps whose Krauss operators are Pauli matrices. A key advantage is that Pauli errors either commute or anti-commute with every stabilizer, so they can be corrected exactly. Besides, Pauli errors are also easy to analyze -- the [various definitions to measure the strength of noise](https://github.com/paviudes/chflow/wiki/Measures-of-noise-strength) coincide and simulate, due to the Gottesman Knill theorem.

There are two popularly known approximations of a CPTP channel to a Pauli channel. These approximations are designed in a manner that the resulting Pauli channel is close to the original CPTP map under some measure of the noise strength.

1. Twirling a Pauli channel: In the Pauli Liouville representation and the Chi matrix representation, a Pauli channel corresponds to a diagonal matrix. For a CPTP map expressed in these representation, _twirling_ a quantum channel corresponds to dropping the off-diagonal entries. In `chflow`, one can obtain the twirled version of a quantum channel using the `chtwirl` command.

```python
>> chan rtz 0.1
Note: the current channel is in the "krauss" representation.
>> chprint
Krauss representation
E_1
[[ 0.951-0.309j  0.000+0.j   ]
 [ 0.000+0.j     0.951+0.309j]]
xxxxxx
>> chtwirl
Twirled channel
	E(R) = 0.904508 R + 0 X R X + 0 Y R Y + 0.0954915 Z R Z.
```

2. Honest Pauli approximation: One of the drawbacks of a Twirling a quantum channel is that it necessarily yields a better quantum channel, i.e, one with a larger Fidelity (or a smaller value of any noise strength measure). Hence, the twirled channel might not be useful in providing upper bounds to error strength (both at the physical as well as the [logical level](https://www.nature.com/articles/srep14670)). An alternate Pauli approximation of a CPTP, that is designed to yield a Pauli channel with a higher noise strength (as per the Trace norm metric) is called the _honest Pauli approximation_. It is defined in terms of an optimization problem with semidefinite constraints. See [this article](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.87.012324) for details. The honest Pauli approximation of a CPTP map can be computed in `chflow` using the `chpa` command.

```python
>> chan rtz 0.1
Note: the current channel is in the "krauss" representation.
>> chpa
Optimization terminated successfully.    (Exit mode 0)
            Current function value: 0.213526188368
            Iterations: 21
            Function evaluations: 91
            Gradient evaluations: 18
  	Optimization completed successfully in 2 seconds.
Honest Pauli Approximation
	E(R) = 0.690982 R + 6.96958e-07 X R X + 6.96958e-07 Y R Y + 0.309016 Z R Z,
	and it has a diamond distance of 0.213526 from the original channel.
>> 
```

Note: The Honest Pauli approximation, though can be used to provide upper bounds on the logical fault rate, these are often found to be extremely loose. See [this article](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.119.040502) for more details.