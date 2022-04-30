# Torcs Environment
from gym_torcs import TorcsEnv
import numpy as np

# Init Logging System
from Utilities.Params import HYPERPARAMS

# RL Algorithms
from Brain.Algorithms import DDPG 

# Create DDPG Agent
actor_path_choosen  = 'Best_Actor_Weights/aggregated_actor.pth'
critic_path_choosen = 'Best Critic_Weights/aggregated_critic.pth'
saved_weights = {'actor_path':actor_path_choosen,
                 'critic_path':critic_path_choosen}
Agent = DDPG(HYPERPARAMS,saved_weights)

# Torcs - Gym 
env = TorcsEnv(vision=False, throttle=True, gear_change=False)

# Training
for i in range(2000):

    if np.mod(i, 3) == 0:
        ob = env.reset(relaunch = True)
    else:
        ob = env.reset()
    
    Agent.speed_X = 0
    s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
    
    total_rewards = 0
    total_damage = 0

    for j in range(100000):

        a_t = Agent.decide(s_t, ob, if_train = False)

        ob, r_t, done, info,damage = env.step(a_t[0])
        Agent.speed_X = ob.speedX * 300
        s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

        s_t = s_t1
        total_rewards = total_rewards + r_t
        total_damage=total_damage+damage

        if done:
            break

    str1 = "Trial : [ {0} ] is completed with reward : [ {1} ] , damage : [ {3} ] and lasted [ {2} ] steps.".format(i+1,total_rewards,j,total_damage)
    print(str1)
    
env.end()
print("Finish.")
