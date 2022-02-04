from brian2 import *
from brian2.equations import refractory
from brian2.monitors import spikemonitor
# import random
import matplotlib.pyplot as plt
import numpy as np


def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(10, 4))
    subplot(121)
    plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(S.i, S.j, 'ok')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')


# This code uses the Brian2 neuromorphic simulator code to implement
#  a version of cyclic shift binding and unbinding based on the 
# paper :High-Dimensional Computing with Sparse Vectors" by Laiho et al 2016. 
# The vector representation is a block structure comprising slots 
# where the number of slots is the vector dimension. In each slot there are a
# number of possible bit positions with one bit set per slot. 
# In this implementation we implement the cyclic shift binding and unbinding 
# operations in Brian2 by representing each slot as a neuron and the time delay
#  of the neuron's spike as the bit position.  

# To ensure that the Brian2 network is performing correctly the first section of the code 
# computes the expected sparse bound vector.  
# The neuromorphic equivalent is implemented as two Brian2 networks.  The first network (net1) implements
# the cyclic binding and the second netwok (net2) implements the cyclic shift unbinding and the clean-up memory
# operation which compares the unbound vector with all the memory vectors to find the best match.
# The sparse bound vector resulting from net1 is passed to net2.

show_bound_vecs_slot_detail = False

# Initialise the network parameters
slots_per_vector = 100  # This is the number of neurons used to represent a vector
bits_per_slot = 100  # This is the number of bit positions
# slots_per_vector = 16  # This is the number of neurons used to represent a vector
# bits_per_slot = 32  # This is the number of bit positions

mem_size = 50  # The number of vectors against which the resulting unbound vector is compared
Num_bound = 5  # The number of vectors that are to be bound

# Time delay between adding cyclically shifted vectors to construct the bound vector is set to 'bits' milliseconds.
input_delay = bits_per_slot  # Time between shifted vectors to construct the bound vector, 'bits' milliseconds.

# NB all timings use milliseconds and we can use a random seed if required.
# np.random.seed(123321)

y_low = 0  # This is used to select the lowest index of the range of neurons that are to be displayed
y_high = 3  # This is used to select the highest index of the range of neurons that are to be displayed

delta = (Num_bound + 1) * bits_per_slot  # Time period over which the Brian2 simulation is to be run.



# Generate a random matrix (P_matrix) which represents all of the sparse vectors that are to be used.
# This matrix has columns equal to the number of slots in each vector with the number of rows equal to the memory size (mem_size)
P_matrix = np.random.randint(0, bits_per_slot, size=(mem_size, slots_per_vector))
# print(P_matrix)
for n in range(0, Num_bound):
    print(P_matrix[n])
print()

# --------------------------------------------------------------------------------------------------------------
# This section of the code computes the theoretical values for the sparse vector (which can then be compared with
# the output of the net1 neuromorphic circuit. It then computes the expected number of bits that will align in the
# clean-up memory operation (which can then be compared with the net2 neuromorphic circuit output).

# print the cyclically shifted version of the vectors that are to be bound
for n in range(0, Num_bound):
    print(np.roll(P_matrix[n], n))

# Create sparse bound vector (s_bound) of zeros
# Init sparse bound vector (s_bound) with zeros
s_bound = np.zeros((slots_per_vector, bits_per_slot))

# Create sparse representation of the bound vector
# - i.e. read the bit position for the shifted vector from the rows of the P_matrix
# and add the vectors together to get s_bound
for n in range(0, Num_bound):
    for s in range(0, slots_per_vector):
        # elementwise cyclic-shift of input vector by bind-order position 'n'
        b = np.roll(P_matrix[n], n)[s]  # Get the posn. of the set bit in the adjacent ('right-shifted') slot.
        s_bound[s][b] += 1  # Add a '1' to the output/bound vector in the current (un-shifted) slot.

# Make s_bound sparse using the argmax function which finds the bit position with the highest random value.
np.set_printoptions(threshold=24)
np.set_printoptions(edgeitems=11)
print("\nResultant Sparse vector, value indicates 'SET' bit position in each slot. "
      "\n(Note, a value of '0' means bit zero is set).\n")

sparse_bound = np.array([np.argmax(s_bound[s]) for s in range(0,slots_per_vector)])
print(sparse_bound)
print()
np.set_printoptions()

# The following unbinds the sparse_bound vector and compare with each of the vectors in the P_matrix couting the
# number of slots that have matching bit positions. This gives the number of spikes that should line up
# in the clean up memory operation.

results = None
miss_matches = []
min_match = slots_per_vector
for nn in range(0, Num_bound):
    results = []
    for test_vec in P_matrix:
        reverse_shifted_vec = np.roll(sparse_bound, -nn)
        unbound_vals = np.array([reverse_shifted_vec[ss] for ss in range(0, slots_per_vector)])
        match = np.count_nonzero(unbound_vals == test_vec)
        results.append(match)

    print(f"source_vec_idx[{nn}], best match idx at vec posn[{np.argmax(results):2d}]:  {results}")

# ---------------------------------------------------------------------------------------------------------------

# This section of the code implements the cyclic shift binding in the Brian2 network (net1)

net1 = Network()

# We first create an array of time delays which will be used to select the first Num_bound vectors from
# the P_matrix with a time delay (input_delay) between each vector.

array1 = np.ones(mem_size) * slots_per_vector * bits_per_slot

for b in range(0, Num_bound):
    array1[b] = (Num_bound - b - 1) * input_delay

#    print (array1[b])


# We use the array1 timedelay matrix to trigger a SpikeGeneratorGroup of neurons that generates the
# required spike triggers and add this to the network.


P = SpikeGeneratorGroup(mem_size, np.arange(mem_size), (array1) * ms)

net1.add(P)

# We now define the set of equation and reset definitions that will be used to generate the neuron action
# potentials and spike reset operations.  Note that we make use of the Brian2 refractory operation.

equ1 = '''
dv/dt = -v/tau : 1 (unless refractory)
vt : 1
tau : second
'''

equ2 = '''
dv/dt = (I)/tau : 1 (unless refractory)
vt :1
I : 1
tau : second
'''

reset1 = '''
vt += v
v=0.0
'''

# The G2 neuron group are the neurons that generate the sparse vectors tht will be bound. To do this each neuron represents
# one slot of the sparse vector and the synaptic connections (S2) on the dendrite represent the time delay of the corresponding spike.
# The time delays are obtained from the P_matrix (S2.delay). The input to this part of the neuromorphic circuit are the 
# sequence of spikes from the 'P' spike generator group. A 'P' spike excites an axon which is connected to all the G2 neurons 
# (S2.connect).  
G2 = NeuronGroup(slots_per_vector, equ1,
                 threshold='v >= 1.0', reset='v=v', method='euler', refractory='t >=((Num_bound)*bits_per_slot)*ms')

G2.v = 0.0
G2.tau = 0.5 * ms

net1.add(G2)
S2 = Synapses(P, G2, 'w : 1', on_pre='v += 1.25')
range_array1 = range(0, slots_per_vector)
for n in range(0, mem_size):
    S2.connect(i=n, j=range_array1)
S2.delay = np.reshape(P_matrix, mem_size * slots_per_vector) * ms

net1.add(S2)

# To perform the cyclic shift and superposition operations the output from G2 is recurrently fed back such that the output from neuron_0 
# feeds to the input of neuron_1 etc. Because Brian2 introduces a time delay of 0.1ms when performing this operation the delay for this
# feedback is the input_delay minus 0.1ms (S3.delay)
S3 = Synapses(G2, G2, 'w : 1', on_pre='v +=1.25')
for n in range(0, slots_per_vector):
    S3.connect(i=n, j=(n + 1) % (slots_per_vector))
S3.delay = (input_delay - 0.1) * ms

net1.add(S3)

# The resulting vector from the recurrent superposition operation is a dense vector. 
# To create the corresponding sparse vector the G4 and G5 neuron groups work together to perform the Argmax operation.
# The G4 neurons perform part of this operation by using a variable spike threshold such that if spikes from the superposed vectors
# have the same time delay then they will only exceed the threshold if the same number of aligne spikes has not occured earlier.
# The G5 neurons then use a linear decaying neuron potential to create a single spike per slot which is the required sparse bound vector.


G4 = NeuronGroup(slots_per_vector, model=equ1, reset=reset1, threshold='v>=vt', method='euler',
                 refractory='t <=((Num_bound-1)*bits_per_slot)*ms')
G4.v = 1.0
G4.vt = -0.5
G4.tau = 0.25 * ms

net1.add(G4)

S4 = Synapses(G2, G4, 'w : 1', on_pre='v +=1.0')
S4.connect(j='i')

net1.add(S4)

G5 = NeuronGroup(slots_per_vector, equ2,
                 threshold='v <= 0.0', reset='v=1.0', method='euler',
                 refractory='t< (Num_bound-1)*bits_per_slot*ms or t> (Num_bound)*bits_per_slot*ms')
G5.v = 0.0
G5.I = -1.0
G5.tau = bits_per_slot * ms

net1.add(G5)

argmax_synapse = Synapses(G4, G5, 'w : 1', on_pre='v = 1.0')
argmax_synapse.connect(j='i')

net1.add(argmax_synapse)

# The following spike and state monitors are defined.

SMP = SpikeMonitor(P)
net1.add(SMP)
M2 = StateMonitor(G2, 'v', record=True)
net1.add(M2)
SM2 = SpikeMonitor(G2)
net1.add(SM2)
SM4 = SpikeMonitor(G4)
net1.add(SM4)
M4 = StateMonitor(G4, 'vt', record=True)
net1.add(M4)
M5 = StateMonitor(G4, 'v', record=True)
net1.add(M5)
SM5 = SpikeMonitor(G5)
net1.add(SM5)
M6 = StateMonitor(G5, 'v', record=True)
net1.add(M6)

# Network 1 is now run for delta milliseconds. 

net1.run(delta * ms)

# Obtain the sparse vector timings from the SM5 monitor and print the timings so that they can be compared with the theoretical values.
array2 = np.array([SM5.i, SM5.t / ms])
sub_array2 = array2[0:2, slots_per_vector:]
print()
sorted_sub_array2 = sub_array2[:, sub_array2[0].argsort()].astype(int) - Num_bound * bits_per_slot
P1_timing = sorted_sub_array2[1]

print(P1_timing)

print()
# The following plots output from the different monitors
subplot(4, 2, 1)
plot(SMP.t / ms, SMP.i, '|')
xlabel('Time (ms)')
ylabel('P Neuron id')
# plt.ylim(0,10)
# plt.xlim(0,2*bits_per_slot*Num_bound)
# plt.xlim(9700,10800)

subplot(4, 2, 2)
plot(SM2.t / ms, SM2.i, '|')
xlabel('Time (ms)')
ylabel('G2 Neuron id')
# plt.xlim(0,2*bits_per_slot*Num_bound)
# plt.ylim(y_low,y_high)
# plt.xlim(690,810)


subplot(4, 2, 3)
plot(M4.t / ms, M4.vt.T)
xlabel('Time (ms)')
ylabel('G4 Threshold Voltage')
# plt.xlim(0,2*bits_per_slot*Num_bound)
# plt.xlim(690,810)

subplot(4, 2, 4)
plot(M5.t / ms, M5.v.T)
xlabel('Time (ms)')
ylabel('G4 Neuron Voltage')
# plt.xlim(0,2*bits_per_slot*Num_bound)
# plt.xlim(690,810)


subplot(4, 2, 5)
plot(SM4.t / ms, SM4.i, '|')
xlabel('Time (ms)')
ylabel('G4 Neuron id')
# plt.xlim(0,2*bits_per_slot*Num_bound)
# plt.ylim(y_low,y_high)
# plt.xlim(690,810)
# plt.ylim(29,31)

subplot(4, 2, 6)
plot(M6.t / ms, M6.v.T)
xlabel('Time (ms)')
ylabel('G5 Neuron Voltage')
# plt.xlim(0,22000)
# plt.xlim(0,2*bits_per_slot*Num_bound)
# plt.xlim(0,2*bits_per_slot*Num_bound)
# plt.xlim(690,900)

subplot(4, 2, 7)
plot(SM5.t / ms, SM5.i, '|')
xlabel('Time (ms)')
ylabel('G5 Neuron id')
# plt.xlim(0,22000)
# plt.xlim(0,2*bits_per_slot*Num_bound)
# plt.xlim(0,2*bits_per_slot*Num_bound)
# plt.ylim(y_low,y_high)
# plt.xlim(790,900)

show()

# --------------------------------------------------------------------------------------------------------
# This section of the code implements the Brian2 neuromorphic circuit which unbinds the vector and then compares the
# unbound vector with each vector in the memory to find the best match (i.e. the clean-up memory operation)

# We first generate the time delay data_matrix which will be used in the 'clean-up memory'  so that the input vector 
# time delay in each slot plus the delay matrix line up at the number of bits per slot 
# (e.g. a time delay in slot 0 of the input vector of say 10 will have a corresponding delay of 90 in the corresponding
#  data_matrix so that if this vector is received then the match condition is an input potential to the neuron at 100)

data_matrix = bits_per_slot - P_matrix

net2 = Network()

# To pass the sparse vector from Net1 into Net2 we create a SpikeGeneratorGroup  that uses the P1_timing from Net1 to generate
# the sparse bound vector which is the input to NeuronGroup G6 (S6).
P1 = SpikeGeneratorGroup(slots_per_vector, np.arange(slots_per_vector), P1_timing * ms)

net2.add(P1)

equ3 = '''
dv/dt = -v / tau : 1 
tau : second
ts:second
'''
# The G6 NeuronGroup performs the recurrent unbinding of the bound sparse vector in this case Neuron_1 connects to Neuron_0 etc. (S7.connect)
# Again the recurrent delay depends on the number of bit positions.
G6 = NeuronGroup(slots_per_vector, equ3, threshold='v >= 0.5', reset='v=0.0', method='euler',
                 refractory='t>= (Num_bound-1)*bits_per_slot*ms')
G6.v = 0.0
G6.tau = 1 * ms

net2.add(G6)

S6 = Synapses(P1, G6, 'w : 1', on_pre='v = 1.0')
S6.connect(j='i')

net2.add(S6)

# Recursively unbind with a delay of 'input_delay' between unbound vectors

S7 = Synapses(G6, G6, 'w : 1', on_pre='v = 1.0')
for n in range(0, slots_per_vector):
    S7.connect(i=n, j=(n - 1) % slots_per_vector)

S7.delay = (input_delay) * ms

net2.add(S7)

# The unbound vector is fed directly into NeuronGroup G7 which performs the 'Clean-Up' memory operation by comparing it in parallel 
# with all of the vectors in the memory.  This operation relies on the alignemnt of the unbound vector spikes delayed by the transpose 
# of the data_matrix (S8.delay). N.B. we have used the predicted min_match to set the threshold for the clean-up memory and just outout
# the index of the best matching vector. Also note that the output order is the reverse of the input order.

G7 = NeuronGroup(mem_size, equ3, threshold='v > min_match-2.0', reset='v=0.0', method='euler')

G7.v = 1.0
G7.tau = 1.0 * ms

net2.add(G7)

range_array1 = range(0, mem_size)
S8 = Synapses(G6, G7, on_pre='v += 1.0')

for n in range(0, slots_per_vector):
    S8.connect(i=n, j=range_array1)

data_matrix2 = np.transpose(data_matrix)
S8.delay = np.reshape(data_matrix2, mem_size * slots_per_vector) * ms

net2.add(S8)

# Create the required monitors

SMP1 = SpikeMonitor(P1)
net2.add(SMP1)

SM6 = SpikeMonitor(G6)

net2.add(SM6)

SM7 = SpikeMonitor(G7)

net2.add(SM7)

M7 = StateMonitor(G7, 'v', record=True)

net2.add(M7)

# Run Network2 for delta milliseconds

net2.run(delta * ms)

# Plot the sparse bound vector

plot(SMP1.t / ms, SMP1.i, '|')
xlabel('Time (ms)')
ylabel('P Neuron id')
show()

# Plot the other monitors

subplot(3, 1, 1)
plot(SM6.t / ms, SM6.i, '|')
xlabel('Time (ms)')
ylabel('G6 Neuron id')
# plt.xlim(0,2*bits_per_slot*Num_bound)
# plt.xlim(bits_per_slot*Num_bound-100,2*bits_per_slot*(Num_bound+1))
# plt.ylim(y_low,y_high)

subplot(3, 1, 2)
plot(M7.t / ms, M7.v.T)
xlabel('Time (ms)')
ylabel('G7 Neuron Voltage')
# plt.xlim(0,2*bits_per_slot*Num_bound)
# plt.xlim(bits_per_slot*Num_bound-100,2*bits_per_slot*(Num_bound+1))

subplot(3, 1, 3)
plot(SM7.t / ms, SM7.i, '|')
xlabel('Time (ms)')
ylabel('G7 Neuron id')

show()
