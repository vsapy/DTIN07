from brian2 import *
from brian2.equations import refractory
from brian2.monitors import spikemonitor
import random
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
    show()

#This code uses the Brian2 neuromorphic simulator code to implement
#  a version of role/filler binding and unbinding based on the 
# paper :High-Dimensional Computing with Sparse Vectors" by Laiho et al 2016. 
# The vector representation is a block structure comprising slots 
# where the number of slots is the vector dimension. In each slot there are a
# number of possible bit positions with one bit set per slot. 
# In this implementation we implement the role/filler binding and unbinding 
# operations in Brian2 by representing each slot as a neuron and the time delay
#  of the neuron's spike as the bit position.  

# To ensure that the Brian2 network is performing correctly the first section of the code 
# computes the expected sparse bound vector.  
# The neuromorphic equivalent is implemented as two Brian2 networks.  The first network (net1) implements
# the role/filler binding and the second netwok (net2) implements the role/filler unbinding and the clean-up memory
# operation which compares the unbound vector with all the memory vectors to find the best match.
# The sparse bound vector resulting from net1 is passed to net2 to initiate the unbinding.

slots_per_vector = 100 # This is the number of neurons used to represent a vector
bits_per_slot = 100  # This is the number of bit positions
mem_size = 1000  # The number of vectors against which the resulting unbound vector is compared
Num_bound = 4  # The number of vectors that are to be bound
input_delay = bits_per_slot  # Time delay between adding cyclically shifted vectors to construct the bound vector is set to 'bits' milliseconds.

# NB all timings use milliseconds and we can use a random seed if required.
# np.random.seed(54321)

target_neuron = 1
y_low=target_neuron-1  # This is used to select the lowest index of the range of neurons that are to be displayed
y_high=target_neuron+1  # This is used to select the highest index of the range of neurons that are to be displayed

delta = (2*Num_bound+2) * bits_per_slot  # This determins the time period over which the Brian2 simulation is to be run.

# Generate a random matrix (P_matrix) which represents all of the sparse vectors that are to be used.
# This matrix has columns equal to the number of slots in each vector with the number of rows equal to the memory size (mem_size)

P_matrix = np.random.randint(0, bits_per_slot, size=(mem_size,slots_per_vector))
Role_matrix = P_matrix[::2]
Val_matrix = P_matrix[1::2]

print(P_matrix)

# Demonstration of 'phase' binding
print(f"\nShowing role*filler bindings for target column {target_neuron}\n")
for n in range(0,2*Num_bound,2):
    print(f"\t{P_matrix[n][target_neuron]}+{P_matrix[n+1][target_neuron]}="
          f"{P_matrix[n][target_neuron]+P_matrix[n+1][target_neuron]} %"
          f" {bits_per_slot} = {(P_matrix[n][target_neuron]+P_matrix[n+1][target_neuron])%bits_per_slot}")
    print()


#--------------------------------------------------------------------------------------------------------------
#
# Theoretical calc
#
# This section of the code computes the theoretical values for the sparse vector (which can then be compared with
# the output of the net1 neuromorphic circuit. It then computes the expected number of bits_per_slot that will align
# in the clean-up memory operation (which can then be compared with the net2 neuromorphic circuit output).
#
# We take pairs of vector and bind them together and in each slot and then store a random
# value between 0 and 1.0 in the slot (this will be used to select just one bit when we create
# the sparse vector)

# Create sparse representation of the bound vector
# Init sparse bound vector (s_bound) with zeros
s_bound = np.zeros((slots_per_vector, bits_per_slot))  # Create a slotted vector with

# Do the binding
for n in range(0, Num_bound):
    for s in range(0, slots_per_vector):  # For each slot
        role_pos = Role_matrix[n][s]   # Position of the set bit in this role vector for this slot
        filler_pos = Val_matrix[n][s]  # Position of the set bit in this value vector for this slot
        b = (filler_pos+role_pos) % bits_per_slot  # Get new 'phase' (bit position) to set in the bound vector's slot
        s_bound[s][b] = rand()

# Make s_bound sparse using the argmax function which finds the bit position with the highest random value.
np.set_printoptions(threshold=24)
np.set_printoptions(edgeitems=11)
print("\nResultant Sparse vector, value indicates 'SET' bit position in each slot. "
      "\n(Note, a value of '0' means bit zero is set).\n")

sparse_bound = np.array([np.argmax(s_bound[s]) for s in range(0,slots_per_vector)])
print(sparse_bound)
print()
np.set_printoptions()


#Unbind the vector sparse_bound vector and compare with each of the vectors in the P_matrix couting the
#number of slots that have matching bit positions. This gives the number of spikes that should line up 
#in the clean up memory operation.

min_match=slots_per_vector
for n in range(0,2*Num_bound,2):    
    for m in range(0,2*Num_bound,2):
        match=0
        for s in range(0, slots_per_vector):
            role_pos = P_matrix[n][s]
            unbound_filler_pos = (sparse_bound[s]-role_pos)%(bits_per_slot)
            if P_matrix[m+1][s] == unbound_filler_pos:
                match +=1
        if n==m:
            print()
            print(n,m,match)
            if match<=min_match:
                    min_match = match
        #When we print the maximum value of match should occur when m=n

print('Min_match=',min_match)


# Generate the time delay data_matrix from the so that the input vector time delay in each slot plus the delay matrix 
# line up at the number of bits per slot (e.g. a time delay in slot 0 of the input vector of say 10 will have a corresponding delay of
# 90 in the corresponding data_matrix so that if this vector is received then the match condition is an input potential to the neuron at 100)


#---------------------------------------------------------------------------------------------------------------

#This section of the code implements the role/filler binding in the Brian2 network (net1)

net1=Network()

#We first create an array of time delays which will be used to select the first Num_bound vectors from 
# the P_matrix with a time delay (input_delay) between each vector.


#Calculate the array for the input spike generator
array1 = np.ones(mem_size) * slots_per_vector * bits_per_slot

# The input spike generator creates pairs of spkies corresponding to contiguous pairs of vectors from the memory that are
# going to be bound together (i.e., vector_0 & vector_1 then vector_2 and Vector_3 etc.)

for b in range(0,2*Num_bound,2):
    array1[b] = (b)*input_delay
    array1[b+1] = (b)*input_delay

# Create the corresponding spike generator group.
P = SpikeGeneratorGroup(mem_size,np.arange(mem_size), (array1)*ms)

net1.add(P)

#We now define the set of equation and reset definitions that will be used to generate the neuron action
#potentials and spike reset operations.  Note that we make use of the Brian2 refractory operation.

equ1 ='''
dv/dt = (I)/tau : 1 
I : 1
tau : second
'''

equ2 = '''
dv/dt = -v/tau : 1 
I : 1
vt : 1
tau : second
'''
reset1 = '''
I=0.0
v=0.0
'''
reset2 = '''
vt = v
v=0.0
'''


# The G1 neurons perform the addition operation in the two selected vectors. Equ1 is a linearly increasing function 
# with a time constant of 2*bits*ms (I=1.0).  The G1 neuron group is stimulated from the P spike generator group with 
# spikes that simultaneously select a role and filler vector using the time delay on the G1 dendrites obtained from the P_matrix (S0.delay)
# On receiving the first spike from either role or filler vector the value of I
# is changed to 0.0 which holds the neuron potential constant until the second spike is received when I = -1.0 and the neuron
#  potential decays until the threshold value v<0.0 when it fires to give the required modulus addition. The value of I is 
# reset to 1.0 using the spike from the P spikemonitorgroup (S1) and the next two vectors are added.

G1 = NeuronGroup(slots_per_vector, equ1,
                 threshold='v < 0.0 or v>=1.0', reset=reset1, method='euler')

G1.v =0.0
G1.I = 1.0
G1.tau = 2 * bits_per_slot * ms

net1.add(G1)


S0 = Synapses(P, G1, 'w : 1',on_pre= 'I = (I-1)')

range_array1 = range(0, slots_per_vector)
for n in range(0,mem_size):
    S0.connect(i=n,j=range_array1)     
S0.delay = np.reshape(P_matrix, mem_size * slots_per_vector) * ms

net1.add(S0)


SP1 = Synapses(P, G1, 'w : 1',on_pre= 'I=1.0')

for n in range(0,mem_size):
    SP1.connect(i=n,j=range_array1)    

net1.add(SP1)



G2 = NeuronGroup(slots_per_vector, equ2, threshold='v>=vt', reset=reset2, method='euler')


G2.v = 0.0
G2.vt=0.1
G2.tau = 1*ms

net1.add(G2)

S12 = Synapses(G1, G2, 'w : 1', on_pre='v += rand()')
S12.connect(j='i') 
net1.add(S12)

G3 = NeuronGroup(slots_per_vector, equ1, threshold='v>1.0 or v<0.0', reset='v=1.0', method='euler')


G3.v = 1.1
G3.I=-1.0
G3.tau = 2 * bits_per_slot * ms

net1.add(G3)

S23 = Synapses(G2, G3, 'w : 1', on_pre='v = 1.0')
S23.connect(j='i') 
net1.add(S23)

# Create the required monitors

SMP = SpikeMonitor(P)
net1.add(SMP)
M1 = StateMonitor(G1, 'v', record=True)
net1.add(M1)
SM1= SpikeMonitor(G1)
net1.add(SM1)
M2 = StateMonitor(G2, 'v', record=True)
net1.add(M2)
SM2= SpikeMonitor(G2)
net1.add(SM2)

M3 = StateMonitor(G3, 'v', record=True)
net1.add(M3)
SM3= SpikeMonitor(G3)
net1.add(SM3)


# Run Net1 for delta milliseconds

net1.run(delta*ms)

# Obtain the sparse vector timings from the SM2 monitor and print the timings so that they can be compared with the theoretical values.
array2 = np.array([SM3.i,SM3.t/ms])
sub_array2 = array2[0:2, slots_per_vector:]
print()
sorted_sub_array2 = sub_array2[:,sub_array2[0].argsort()].astype(int) 
P1_timing = sorted_sub_array2[1]
P1_timing = P1_timing[P1_timing >= 2 * Num_bound * bits_per_slot] - 2 * Num_bound * bits_per_slot
print(P1_timing)

# The following plots output from the different monitors
subplot(7,1,1)
plot(SMP.t/ms, SMP.i,'|')
xlabel('Time (ms)')
ylabel('P Neuron id')

subplot(7,1,2)
plot(M1.t/ms, M1.v[target_neuron].T)
xlabel('Time (ms)')
ylabel('G1 Neuron Voltage')

subplot(7,1,3)
plot(SM1.t/ms, SM1.i,'|')
xlabel('Time (ms)')
ylabel('G1 Neuron id')
plt.ylim(y_low,y_high)

subplot(7,1,4)
plot(M2.t/ms, M2.v[target_neuron].T)
xlabel('Time (ms)')
ylabel('G2 Neuron Voltage')
#plt.xlim(12000,14000)

subplot(7,1,5)
plot(SM2.t/ms, SM2.i,'|')
xlabel('Time (ms)')
ylabel('G2 Neuron id')
plt.ylim(y_low,y_high)

subplot(7,1,6)
plot(M3.t/ms, M3.v[target_neuron].T)
xlabel('Time (ms)')
ylabel('G3 Neuron Voltage')
#plt.xlim(12000,14000)

subplot(7,1,7)
plot(SM3.t/ms, SM3.i,'|')
xlabel('Time (ms)')
ylabel('G3 Neuron id')
plt.ylim(y_low,y_high)

show()

#--------------------------------------------------------------------------------------------------------
#This section of the code implements the Brian2 neuromorphic circuit which unbinds the vector. 
#The unbound vector and a selected role vector are processed to give the corresponding 'noisy' filler vector.
# which is then compared to the memory vectors to find the best match (i.e. the clean-up memory operation)


# We first generate the time delay data_matrix which will be used in the 'clean-up memory'  so that the input vector 
# time delay in each slot plus the delay matrix line up at the number of bits per slot 
# (e.g. a time delay in slot 0 of the input vector of say 10 will have a corresponding delay of 90 in the corresponding
#  data_matrix so that if this vector is received then the match condition is an input potential to the neuron at 100)

data_matrix = bits_per_slot - P_matrix

net2=Network()

print()

# To pass the sparse vector from Net1 into Net2 we create a SpikeGeneratorGroup that uses the P1_timing from Net1 to generate
# the sparse bound vector which is the input to NeuronGroup G6 (S6).

P1 = SpikeGeneratorGroup(slots_per_vector, np.arange(slots_per_vector), P1_timing * ms)

net2.add(P1)

# We now define the neuron potential equations and resets plus a preset
equ2 = '''
dv/dt = -v/tau : 1 
I : 1
tau : second
'''

equ3 ='''
dv/dt = (I)/tau : 1 
I : 1
tau : second
'''

reset3 = '''
I=1.0
v=0.0
'''
preset1 = '''
I = 1.0
v= 0.0
'''

# NeuronGroup G7 is a recurrent circuit which simply repeates the sparse bound vector from P1 every 3*bits milliseconds 
# and feeds the output vector into the G6 neurongroup (see S7 below)

G7 = NeuronGroup(slots_per_vector, equ2, threshold='v>=1.0', reset='v=0.0', method='euler')
G7.v=0.0
G7.tau = 0.5*ms

SP17 = Synapses(P1, G7, 'w : 1',on_pre= 'v=1.25')
SP17.connect(j='i')
#SP17.delay = bits*ms


S77 = Synapses(G7, G7, 'w : 1',on_pre= 'v=1.25')
S77.connect(j='i')
#S77.delay = 3*bits*ms
S77.delay = 2 * bits_per_slot * ms

net2.add(G7)
net2.add(SP17)
net2.add(S77)



#Calculate the array for the input spike generator which cycles through the role vectors 0,2,4 etc
array2 = np.ones(mem_size) * slots_per_vector * bits_per_slot
for b in range(0,Num_bound):
    #array2[b*2] = (b*3)*input_delay
    array2[b*2] = (b*2)*input_delay

P2 = SpikeGeneratorGroup(mem_size,np.arange(mem_size), (array2)*ms)
net2.add(P2)

#The G6 neuron group is stimulated from the P spike generator group with and the G7 neuron group.
#The P spike generator generates a role vector role using the time delay on the G6 dendrites obtained from the P_matrix (S5.delay)
# and the G6 neuron group produces the sparse bound vector.

# The G6 neurons perform the subtraction operation on the selected vectors. In this case Equ3 is a linearly increasing function 
# with a time constant of bits*ms (I=1.0).  On receiving the first spike from either role or filler vector the value of I=0.0
# which holds the neuron potential constant until the second spike is received when I again becomes 1.0  and the neuron
#  potential continues to increase until the threshold value v>1.0 when it fires. To give the required modulus addition the value 
# of I is maintained at 1.0 to ensure a second vector is generated. One of these two vector will have the correct modulus timings and so we compare both vectors in the final 
# neuron group stage (G8) to get the best match. 


G6 = NeuronGroup(slots_per_vector, equ3, threshold='v>=1.0', reset=reset3, method='euler', refractory ='2*Num_bound*ms')

G6.v =0.0
G6.I = 1.0
G6.tau = bits_per_slot * ms

net2.add(G6)



S5 = Synapses(P2, G6, 'w : 1',on_pre= 'I = (I-1)%2')

range_array2 = range(0, slots_per_vector)
for n in range(0,mem_size):
    S5.connect(i=n,j=range_array2)     
S5.delay = np.reshape(P_matrix, mem_size * slots_per_vector) * ms

net2.add(S5)

S6 = Synapses(P2, G6, 'w : 1',on_pre= preset1)

for n in range(0,mem_size):
    S6.connect(i=n,j=range_array2)    

net2.add(S6)


S7 = Synapses(G7, G6, 'w : 1',on_pre= 'I = (I-1)%2')
S7.connect(j='i')



net2.add(S7)

# This final NeuronGroup,G8, stage is the clean up memory operation using the transpose of the data_matrix to set the 
# synaptic delays on the G8 dendrites. We only produce one output spike per match by using the refractory operator to
#  suppress any further spikes. This could be improved to choose the larget matching spike.

G8 = NeuronGroup(mem_size, equ2, threshold='v >= 11.0', reset='v=0.0', method='euler')

G8.v = 1.0
G8.tau = 2.0*ms

net2.add(G8)

range_array3 = range(0,mem_size)

S8 = Synapses(G6, G8, on_pre='v += 1.0')

for n in range(0, slots_per_vector):
    S8.connect(i=n,j=range_array3)  

data_matrix2 = np.transpose(data_matrix) 
S8.delay = np.reshape(data_matrix2, mem_size * slots_per_vector) * ms
net2.add(S8)

# Create the required monitors

SMP1 = SpikeMonitor(P1)

net2.add(SMP1)

SM7 = SpikeMonitor(G7)
net2.add(SM7)

SMP2 = SpikeMonitor(P2)

net2.add(SMP2)

M6 = StateMonitor(G6, 'v', record=True)

net2.add(M6)

SM6 = SpikeMonitor(G6)

net2.add(SM6)


M8 = StateMonitor(G8, 'v', record=True)

net2.add(M8)

SM8 = SpikeMonitor(G8)

net2.add(SM8)

net2.run(((2*Num_bound) * bits_per_slot + 3) * ms)

# Plot the sparse bound vector

plot(SMP1.t/ms, SMP1.i,'|')
xlabel('Time (ms)')
ylabel('P Neuron id')
show()

# Plot the other monitors
subplot(6,1,1)
plot(SM7.t/ms, SM7.i,'|')
xlabel('Time (ms)')
ylabel('P1 Neuron id')
plt.ylim(0, slots_per_vector)


subplot(6,1,2)
plot(SMP2.t/ms, SMP2.i,'|')
xlabel('Time (ms)')
ylabel('P2 Neuron id')
#plt.xlim(0,2*bits*Num_bound)
#plt.xlim(bits*Num_bound-100,2*bits*(Num_bound+1))
#plt.ylim(y_low,y_high)

subplot(6,1,3)
plot(M6.t/ms, M6.v[0].T)
xlabel('Time (ms)')
ylabel('G6 Neuron Voltage')
#plt.xlim(0,2*bits*Num_bound)
#plt.xlim(bits*Num_bound-100,2*bits*(Num_bound+1))

subplot(6,1,4)
plot(SM6.t/ms, SM6.i,'|')
xlabel('Time (ms)')
ylabel('G6 Neuron id')

subplot(6,1,5)
plot(M8.t/ms, M8.v.T)
xlabel('Time (ms)')
ylabel('G8 Neuron Voltage')

subplot(6,1,6)
plot(SM8.t/ms, SM8.i,'|')
xlabel('Time (ms)')
ylabel('G8 Neuron id')

show()
