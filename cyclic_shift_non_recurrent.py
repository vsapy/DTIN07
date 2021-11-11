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

# Initialise the network parameters
slots_per_vector = 500  # This is the number of neurons used to represent a vector
bits_per_slots = 512  # This is the number of bit positions
mem_size = 1000  # The number of vectors against which the resulting unbound vector is compared
Num_bound = 20  # The number of vectors that are to be bound
input_delay = bits_per_slots  # Time delay between adding cyclically shifted vectors to construct the bound vector is set to 'bits' milliseconds.

# NB all timings use milliseconds and we can use a random seed if required.
# np.random.seed(654321)

y_low = 0  # This is used to select the lowest index of the range of neurons that are to be displayed
y_high = slots_per_vector - 1  # This is used to select the highest index of the range of neurons that are to be displayed

delta = (Num_bound) * bits_per_slots  # This determins the time period over which the Brian2 simulation is to be run.

# Generate a random matrix (P_matrix) which represents all of the sparse vectors that are to be used.
# This matrix has columns equal to the number of slots in each vector with the number of rows equal to the memory size (mem_size)
P_matrix = np.random.randint(0, bits_per_slots, size=(mem_size, slots_per_vector))
# print(P_matrix)
'''
for n in range(0,Num_bound):
    print(P_matrix[n])
print()
'''

# This section of the code implements the cyclic shift binding in the Brian2 network (net1)

net1 = Network()

# We first create an array of time delays which will be used to select the first Num_bound vectors from
# the P_matrix with a time delay (input_delay) between each vector.

array1 = np.ones(mem_size) * slots_per_vector * bits_per_slots

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
dv/dt = -v/tau : 1 
tau : second
'''

# The G1 neuron group are the neurons that generate the sparse vectors tht will be bound. To do this each neuron represents
# one slot of the sparse vector and the synaptic connections (SP1) on the dendrite represent the time delay of the corresponding spike.
# The time delays are obtained from the P_matrix (SP1.delay). The input to this part of the neuromorphic circuit are the 
# sequence of spikes from the 'P' spike generator group. A 'P' spike excites an axon which is connected to all the G1 neurons 
# (SP1.connect).  The output from the group is recursively fed to the next neuron to provide the cyclic shift.  This then gives two 
# possible spikes in the next cycle. Using the refractory property of the neuron only the first of these generates a spike.
G1 = NeuronGroup(slots_per_vector, equ1,
                 threshold='v >= 0.5', reset='v=0.0', method='euler', refractory='t%(bits_per_slot*ms)')

G1.v = 0.0
G1.tau = 1.0 * ms

net1.add(G1)
SP1 = Synapses(P, G1, 'w : 1', on_pre='v = 1.0')
range_array1 = range(0, slots_per_vector)
for n in range(0, mem_size):
    SP1.connect(i=n, j=range_array1)
SP1.delay = np.reshape(P_matrix, mem_size * slots_per_vector) * ms

net1.add(SP1)

# To perform the cyclic shift and superposition operations the output from G2 is recurrently fed back such that the output from neuron_0 
# feeds to the input of neuron_1 etc. Because Brian2 introduces a time delay of 0.1ms when performing this operation the delay for this
# feedback is the input_delay minus 0.1ms (S3.delay)
S11 = Synapses(G1, G1, 'w : 1', on_pre='v +=1.0')
for n in range(0, slots_per_vector):
    S11.connect(i=n, j=(n + 1) % (slots_per_vector))
S11.delay = (input_delay - 0.1) * ms

net1.add(S11)

# The following spike and state monitors are defined.

SMP = SpikeMonitor(P)
net1.add(SMP)
M1 = StateMonitor(G1, 'v', record=True)
net1.add(M1)
SM1 = SpikeMonitor(G1)
net1.add(SM1)

# Network 1 is now run for delta milliseconds.

net1.run(delta * ms)

# Obtain the sparse vector timings from the SM5 monitor and print the timings so that they can be compared with the theoretical values.


array2 = np.array([SM1.i, SM1.t / ms])
sub_array2 = array2[0:2, slots_per_vector:]
print()
print(sub_array2)
sorted_sub_array2 = sub_array2[:, sub_array2[0].argsort()].astype(int)
print()
print(sorted_sub_array2)
print()

P1_timing = sorted_sub_array2[:, sorted_sub_array2[1, :] >= bits_per_slots * (Num_bound - 1)][1] - bits_per_slots * (
            Num_bound - 1)
print(len(P1_timing))
print(P1_timing)

print()
# The following plots output from the different monitors
subplot(3, 1, 1)
plot(SMP.t / ms, SMP.i, '|')
xlabel('Time (ms)')
ylabel('P Neuron id')

subplot(3, 1, 2)
plot(M1[1].t / ms, M1[1].v.T)
xlabel('Time (ms)')
ylabel('G2 Threshold Voltage')

subplot(3, 1, 3)
plot(SM1.t / ms, SM1.i, '|')
xlabel('Time (ms)')
ylabel('G2 Neuron id')
plt.ylim(y_low, y_high)

show()

data_matrix = bits_per_slots - P_matrix

net2 = Network()

# To pass the sparse vector from Net1 into Net2 we create a SpikeGeneratorGroup  that uses the P1_timing from Net1 to generate
# the sparse bound vector which is the input to NeuronGroup G6 (S6).
P1 = SpikeGeneratorGroup(slots_per_vector, np.arange(slots_per_vector), P1_timing * ms)

net2.add(P1)

equ2 = '''
dv/dt = -v / tau : 1 
tau : second
ts:second
'''
# The G2 NeuronGroup performs the recurrent unbinding of the bound sparse vector in this case Neuron_1 connects to Neuron_0 etc. (S22.connect)
# Again the recurrent delay depends on the number of bit positions.
G2 = NeuronGroup(slots_per_vector, equ2, threshold='v >= 0.5', reset='v=0.0', method='euler')
G2.v = 0.0
G2.tau = 1 * ms

net2.add(G2)

SP2 = Synapses(P1, G2, 'w : 1', on_pre='v = 1.0')
SP2.connect(j='i')

net2.add(SP2)

# Recursively unbind with a delay of 'input_delay' between unbound vectors

S22 = Synapses(G2, G2, 'w : 1', on_pre='v = 1.0')
for n in range(0, slots_per_vector):
    S22.connect(i=n, j=(n - 1) % slots_per_vector)

S22.delay = (1 * input_delay) * ms

net2.add(S22)

# The unbound vector is fed directly into NeuronGroup G3 which performs the 'Clean-Up' memory operation by comparing it in parallel 
# with all of the vectors in the memory.  This operation relies on the alignemnt of the unbound vector spikes delayed by the transpose 
# of the data_matrix (S8.delay). N.B. we have used the predicted min_match to set the threshold for the clean-up memory and just outout
# the index of the best matching vector. Also note that the output order is the reverse of the input order.

G3 = NeuronGroup(mem_size, equ2, threshold='v >= 10', reset='v=0.0', method='euler')

G3.v = 1.0
G3.tau = 1.0 * ms

net2.add(G3)

range_array1 = range(0, mem_size)
S23 = Synapses(G2, G3, on_pre='v += 1.0')

for n in range(0, slots_per_vector):
    S23.connect(i=n, j=range_array1)

data_matrix2 = np.transpose(data_matrix)
S23.delay = np.reshape(data_matrix2, mem_size * slots_per_vector) * ms

net2.add(S23)

# Create the required monitors

SMP1 = SpikeMonitor(P1)
net2.add(SMP1)

SM2 = SpikeMonitor(G2)

net2.add(SM2)

SM3 = SpikeMonitor(G3)

net2.add(SM3)

M3 = StateMonitor(G3, 'v', record=True)

net2.add(M3)

# Run Network2 for delta milliseconds

net2.run((delta + bits_per_slots) * ms)

# Plot the sparse bound vector

plot(SMP1.t / ms, SMP1.i, '|')
xlabel('Time (ms)')
ylabel('P Neuron id')
show()

# Plot the other monitors

subplot(3, 1, 1)
plot(SM2.t / ms, SM2.i, '|')
xlabel('Time (ms)')
ylabel('G2 Neuron id')
# plt.ylim(y_low,y_high)

subplot(3, 1, 2)
plot(M3.t / ms, M3.v.T)
xlabel('Time (ms)')
ylabel('G3 Neuron Voltage')

subplot(3, 1, 3)
plot(SM3.t / ms, SM3.i, '|')
xlabel('Time (ms)')
ylabel('G3 Neuron id')

show()
