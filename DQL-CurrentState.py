import math
import numpy as np
from scipy import signal
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, InputLayer,  LSTM


def gaussmf(x,sigma,c):
    f = math.exp(-math.pow(x-c,2)/(2*math.pow(sigma,2)))
    return f



def Func_of_Cal_reward_two_AP(TTT_LV,TTT_VL,k,t):
    Tex = 0.8
    #math.p
    # VLC AP parameters
    # semi - angle at half power
    theta = 60
    # Lambertian order of emission
    ml = -math.log10(2) / math.log10(math.cos(math.radians(theta)))
    # transmitted optical power by a VLC AP
    P_VLC_AP = 1
    # detector physical area of a PD
    Adet = 1e-4
    # gain of an optical filter; ignore if no filter is used
    Ts = 1
    # refractive index of a lens at a PD; ignore if no lens is used
    index = 1.5
    # FOV of a receiver
    FOV = 90
    # gain of an optical concentrator; ignore if no lens is used
    G_Con = math.pow(index, 2) / math.pow(math.sin(math.radians(FOV)), 2)
    # O / E conversion efficiency
    OE_eff = 0.53
    # User and VLC APs location
    # VLC APs location
    VLC1_loc = [-1.5, 0, 2.4]
    VLC2_loc = [1.5, 0, 2.4]
    # Number of users
    Num_of_user = round(1200/t * gaussmf(k, t/4, t/2))
    if Num_of_user <= 0:
        Num_of_user = 1
    # Uniform random number pool
    random.seed(42)
    N = Num_of_user+10  # no of samples
    x = range(N)
    rand_pool = [random.uniform(0,1) for i in x]
    reward_user = np.zeros((Num_of_user,1))
    #load rand_pool.mat
    for x in range(0,Num_of_user):
        # UE moving speed is time dependent
        UE_speed_mean = 1.5 - gaussmf(k, t / 4, t / 2)
        a = UE_speed_mean - 0.2
        b = UE_speed_mean + 0.2
        UE_speed = a + (b - a)* rand_pool[x]
        #SINR report interval 10  ms
        SINR_report_interval = 0.01
        # Candidate UE positions
        #UE_loc = []
        #h, w = 3, 10
        #UE_loc = [[0 for x in range(w)] for y in range(h)]
        UE_loc = np.zeros((len(np.arange(-4, 4, UE_speed * SINR_report_interval)), 3))
        UE_loc[:,0] = np.arange(-4, 4, UE_speed * SINR_report_interval)
        UE_loc[:, 1] = 0 #np.zeros((len(UE_loc),1)) #repmat(0, length(UE_loc), 1);
        UE_loc[:, 2] =1 # np.ones((len(UE_loc),1)) #repmat(1, length(UE_loc), 1);
        # LTE and VLC bandwidth & LTE SINR
        # LTE and VLC bandwidth are both 20 MHz
        W_L = 20e6 * (1.1 - gaussmf(k, t / 4, t / 2))
        W_V = 20e6
        # LTE SINR is 20 dB
        LTE_SINR = 20
        # Calculate the RSS % Noise in unit A ^ 2
        Noise = 4.7e-14
        # Distance between VLC AP and UE
        #VLC1_D = sqrt((VLC1_loc(1) - UE_loc(:, 1)). ^ 2 + (VLC1_loc(2) - UE_loc(:, 2)).^ 2 + (VLC1_loc(3) - UE_loc(:, 3)).^ 2);
        VLC1_D = np.sqrt(np.power((VLC1_loc[0] - UE_loc[:, 0]), 2) + np.power((VLC1_loc[1] - UE_loc[:, 1]), 2) + np.power((VLC1_loc[2] - UE_loc[:, 2]), 2))
        VLC2_D = np.sqrt(np.power((VLC2_loc[0] - UE_loc[:, 0]), 2) + np.power((VLC2_loc[1] - UE_loc[:, 1]), 2) + np.power((VLC2_loc[2] - UE_loc[:, 2]), 2))
        #VLC2_D = math.sqrt(VLC2_loc[0] - math.pow(UE_loc[:, 0], 2) + (VLC2_loc[1] - math.pow(UE_loc[:, 1]), 2) + math.pow(VLC2_loc[2] - UE_loc[:, 2]), 2)
        #VLC2_D = sqrt((VLC2_loc(1) - UE_loc(:, 1)). ^ 2 + (VLC2_loc(2) - UE_loc(:, 2)).^ 2 + (VLC2_loc(3) - UE_loc(:, 3)).^ 2);
        VLC1_cosphi = (VLC1_loc[2] - UE_loc[:, 2])/ VLC1_D
        VLC2_cosphi = (VLC2_loc[2] - UE_loc[:, 2])/ VLC2_D
        # Channel gains of VLC APs
        VLC1_H = (ml + 1) * Adet * np.power(VLC1_cosphi, ml) / (2 * math.pi * np.power(VLC1_D, 2) * VLC1_cosphi)
        VLC2_H = (ml + 1) * Adet * np.power(VLC2_cosphi, ml) / (2 * math.pi * np.power(VLC2_D, 2) * VLC2_cosphi)
        #VLC2_H = (ml + 1) * Adet. * VLC2_cosphi. ^ (ml). / (2 * pi. * VLC2_D. ^ 2). * VLC2_cosphi;
        # RSS values
        VLC1_RSS = P_VLC_AP * VLC1_H
        VLC2_RSS = P_VLC_AP * VLC2_H
        # Signal - to - noise ratio values
        xx = 12
        yy = 3
        zz=0.01
        VLC1_SNR = 10 * np.log10(np.power(VLC1_RSS * OE_eff, 2) / Noise)
        # Signal - to - interference - plus - noise ratio values
        VLC1_SINR = 10 * np.log10(np.power(VLC1_RSS * OE_eff,  2) / (Noise + np.power(VLC2_RSS * OE_eff, 2)))
        VLC2_SINR = 10 * np.log10(np.power(VLC2_RSS * OE_eff, 2) / (Noise + np.power(VLC1_RSS * OE_eff, 2)))
        #VLC2_SINR = 10 * math.log10((VLC2_RSS * OE_eff) ^ 2. / (Noise + (VLC1_RSS. * OE_eff). ^ 2));
        VLC_SINR = np.maximum(VLC1_SINR,VLC2_SINR)
        # conn = 0 for LTE and conn = 1 for VLC
        conn = 0
        i = 0
        #print(TTT_LV)
        #print(type(TTT_LV))
        count_LV = TTT_LV/0.01# [xx / 0.01 for xx in TTT_LV]
        #print(count_LV)
        count_VL = TTT_VL/0.01 #[xx / 0.01 for xx in TTT_VL]
        T = np.zeros((len(VLC_SINR), 1))
        while i < len(VLC_SINR):
            if conn == 0 and VLC_SINR[i] < 10:
                T[i] = W_L * math.log2(1 + math.pow(10, (LTE_SINR / 10)))
                count_LV = TTT_LV/0.01  #[xx / 0.01 for xx in TTT_LV]
                i = i + 1
            elif  conn == 0 and VLC_SINR[i] >= 10 and count_LV > 0:
                T[i] = W_L * math.log2(1 + math.pow(10, (LTE_SINR / 10)))
                count_LV = count_LV -1 # [xx - 1 for xx in count_LV]
                i = i + 1
            elif conn == 0 and VLC_SINR[i] >= 10 and count_LV == 0:
                T[i: int(i + Tex/0.01)] = 0
                count_LV = TTT_LV/0.01  #[xx / 0.01 for xx in TTT_LV]
                conn = 1
                i = int(i + Tex / 0.01)
            elif conn == 1 and VLC_SINR[i] >= 10:
                T[i] = W_V * math.log2(1 + math.pow(10, (VLC_SINR[i] / 10)))
                count_VL = TTT_VL/0.01  #[xx / 0.01 for xx in TTT_VL]
                i = i + 1
            elif conn == 1 and VLC_SINR[i] < 10 and count_VL > 0:
                T[i] = W_V * math.log2(1 + math.pow(10, (VLC_SINR[i] / 10)))
                count_VL = count_VL-1 # [xx - 1 for xx in count_VL]
                i = i + 1
            elif conn == 1 and VLC_SINR[i] < 10 and count_VL == 0:
                T[i: int(i + Tex / 0.01)] = 0
                count_VL = TTT_VL/0.01 #[xx/0.01 for xx in TTT_VL]
                conn = 0
                i = int(i + Tex / 0.01)
            else: print('ERROR')
        reward_user[x] = np.mean(T) / 1e6
    reward = np.mean(reward_user)
    return reward


# Create the matrix of TTT combination
#16 different TTT values for LTE-to-VLC and VLC-to-LTE
TTT_LV =  [0.0, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12]
TTT_VL = TTT_LV
N = len(TTT_LV)*len(TTT_VL)
n = math.sqrt(N)
print(n)
# Create the matrix 16x16, each column corresponds to one possible value of
# TTT_LV and each row corresponds to one possible value of TTT_VL
TTT_comb = np.zeros((N,2))
for i in range(1,N+1):
    q = int(i/n)
    r = int(i%n)
    if r!=0:
        q=q+1
    else:
        r=int(n)
    TTT_comb[i-1,:] = [TTT_LV[q-1], TTT_VL[r-1]]


# Create reward matrix for limiting the possible actions at each state (TTT combination)
# We have 9 possible actions:
#
# Increase TTT_LV by a single level: (i+1)
# Decrease TTT_LV by a single level: (i-1)
# Increase TTT_VL by a single level: (i-n)
# Decrease TTT_VL by a single level: (i+n)
# Increase both TTT_LV and TTT_VL by a single level: (i+n+1)
# Decrease both TTT_LV and TTT_VL by a single level: (i-n-1)
# Increase TTT_LV and decrease TTT_VL by a single level: (i-n+1)
# Increase TTT_VL and decrease TTT_LV by a single level: (i+n-1)
# No change to the current values of TTT_LV and TTT_VL: i

model = Sequential()
#model.add(InputLayer(batch_input_shape=(1,81)))
model.add(LSTM(100,input_shape=(1,81)))
model.add(Dense(81, activation='relu'))
#model.add(Dense(81, activation='relu'))
#model.add(Dense(81, activation='relu'))
#model.add(Dense(81, activation='relu'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])



count=0
T = [12, 24, 48, 96] # [24]
maxItr = 1000
Throughput_Average = [[0 for x in range(maxItr)] for y in range(len(T))]
for t in T:
    print("The T is "+str(t))
    Current = [[0 for x in range(t)] for y in range(20)]
    #reward=np.ones((N,N,t))
    reward = np.ones((t, N, N))
    for i in range(0,N):
        for j in range(0,N):
            for k in range(0,t):
                if j!=i+1  and j!=i-1  and j!=i-n and j!=i+n and j!=i+n+1 and j!=i-n-1 and j!=i-n+1 and j!=i+n-1 and j!=i:
                    #reward[i][j][k] = float('-inf')
                    reward[k][i][j] = float('-inf')
    for i in range(0, N, int(n)):
        for j in range(0, int(i+n)):
            for k in range(0,t):
                if j == i + n - 1 or j == i - 1 or j == i - n - 1:
                    #reward[i][j][k] = float('-inf')
                    #reward[j][i][k] = float('-inf')
                    reward[k][i][j] = float('-inf')
                    reward[k][j][i] = float('-inf')
    for i in range(0,N):
        for j in range(0,N):
            for k in range(1,t+1):
                #if reward[i][j][k] >0:
                if reward[k-1][i][j] > 0:
                    #reward[i][j][k] = Func_of_Cal_reward_two_AP(TTT_comb[j, 0], TTT_comb[j, 1], k, t)
                    reward[k-1][i][j] = Func_of_Cal_reward_two_AP(TTT_comb[j, 0], TTT_comb[j, 1], k, t)
                    #print(k,t,i,j)
                    #print(reward[i][j][k])
                #else: print("ELSE", reward[i][j][k])
    #filename = sprintf('t%dsmall.mat', t);
    #save(filename);
    #disp(t)
    # Q - learning algorithm
    # Initialize the Q - table with random values ~ N(0, 1)
    # Set learnning rate to 1 and discount factor to 0.9
    # The maximum number of episodes is set to 50
    #reward = np.transpose(reward)
    #q= np.random.randn(reward.shape[0], reward.shape[1], reward.shape[2])
    gamma = 0.9
    alpha = 1
    #maxItr = 50 #### 10000
    epsilon_initial = 0.2
    #cs -> current state
    # ns -> next state
    # Repeat until Convergence OR Maximum Iterations
    for i in range(0, maxItr):
        epsilon = epsilon_initial / (1 + i / 500)
        # Starting from start position
        cs =73
        # Repeat for t times
        Throughput = [0 for i in range(0, t)]
        for k in range(0,t):
            # possible actions for the chosen state
            n_actions = np.where(reward[k][cs][:] >= 0)

            if i < 20:
                Current[i][k] = cs
                #print(cs)
            #n_actions = np.where(reward[k][:][cs] >= 0)
            #print(n_actions)
            # choose an action at random with probability epsilon and set it as the
            # next state
            if np.random.rand(1) < epsilon:
                ns = n_actions[0][np.random.randint(len(n_actions[0]))]
                #print(ns)
            else:
                #print(n_actions)
                #print(k)
                #print(cs)
                #print(q[k][n_actions][cs])
                #print(np.max(q[k][n_actions][cs]))
                ###########################################
                qtable = model.predict(np.identity(81)[cs:cs + 1].reshape(1,1,81))[0]
                maxinlist = [qtable[ii] for ii in n_actions[0]]
                ##########################################
                #maxinlist = [q[k][ii][cs] for ii in n_actions[0]]
                #print(maxinlist)
                m = max(maxinlist)
                mm = [ii for ii, jj in enumerate(maxinlist) if jj == m]
                ns = [n_actions[0][ii] for ii in mm]
                #ns = n_actions[0][maxinlist.index(np.max(maxinlist))]
                #ns = n_actions[np.where(q[k][n_actions][cs]==np.max(q[k][n_actions][cs]))]
                if len(ns)>1:
                    ns = ns[np.random.randint(len(ns))]
                else: ns = ns[0]
            if k < t-1:
                # find all the possible actions for the selected state
                n_actions = np.where(reward[k + 1][ns][:] >= 0)
                #n_actions = np.where(reward[k+1][:][ns] >= 0)
                #n_actions.item(0)
                # find the maximum q - value i.e, next state with best action
                max_q = 0
                qtable = model.predict(np.identity(81)[ns:ns + 1].reshape(1,1,81))[0]
                #ns = ns[0]
                for j in range(0,len(n_actions[0])):
                    #max_q = max(max_q, q(ns, n_actions(j), k + 1));
                    max_q = np.maximum(max_q, qtable[n_actions[0][j]])
                    #max_q = np.maximum(max_q, q[k+1][n_actions[0][j]][ns])
                # Update q- values as perellman's equation
                #for nss in ns:
                target = reward[k][cs][ns] + gamma * max_q
                #target = reward[k][ns][cs] + gamma * max_q
                qtable[ns]= target
                #qtable2 = qtable.reshape(-1,81)
                model.fit(np.identity(81)[ns:ns+1].reshape(1,1,81), qtable.reshape(-1,81), epochs=1, verbose=0)
                #q[k][ns][cs] = reward[k][ns][cs] + gamma * max_q
            else:
                #for nss in ns:
                qtable = model.predict(np.identity(81)[ns:ns + 1].reshape(1,1,81))[0]
                target = reward[k][cs][ns]
                #target = reward[k][ns][cs]
                qtable[ns]= target
                model.fit(np.identity(81)[ns:ns + 1].reshape(1,1,81), qtable.reshape(-1, 81), epochs=1, verbose=0)
                #q[k][ns][cs] = reward[k][ns][cs]
            # Set current state as next state
            cs = ns
            Throughput[k] = Func_of_Cal_reward_two_AP(TTT_comb[cs,0], TTT_comb[cs,1], k+1, t)
            #print(Throughput)
        #print("")
        Throughput_Average[count][i] = np.mean(Throughput)
    print(Current)
    count = count + 1

plt.plot(range(0,maxItr), Throughput_Average[:][0], label = "T=12")
plt.plot(range(0,maxItr), Throughput_Average[:][1], label = "T=24")
plt.plot(range(0,maxItr), Throughput_Average[:][2], label = "T=48")
plt.plot(range(0,maxItr), Throughput_Average[:][3], label = "T=96")

plt.ylabel('Average throughput (Mbps)')
plt.xlabel('Episode index')
plt.legend()
plt.show()
