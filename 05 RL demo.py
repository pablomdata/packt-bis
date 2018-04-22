import gym
import numpy as np
from tqdm import tqdm
from time import sleep

def featurize(s,a):
    return (2*a-1)*s

def Q(s,a, theta):
    vec = featurize(s,a)
    return np.dot(vec,theta)

env = gym.make("CartPole-v0")
actions = range(env.action_space.n)
theta_star = np.array([0,0,3,1])
gamma = 0.99
n_episodes = 10000
alpha = 1e-3


np.random.seed(42)
theta = np.array([-0.1,0.1,2.5,1.1])
theta = np.random.rand(4)

print("*"*100)
print("Training...")
for ep in tqdm(range(n_episodes)):
    state = env.reset()
    ep_reward = 0
    done = False
    step = 0
    while not done:
        step +=1
        action = np.argmax([Q(state,a,theta) for a in actions]) if np.random.random()>10/(ep+1) else np.random.choice(actions)
        new_state, reward, done, _ = env.step(action)
        td_target = reward + gamma*np.max([Q(new_state,a,theta) for a in actions])
        theta = theta - alpha*((td_target - Q(state,action, theta)))*featurize(state,action)
        state = new_state

        ep_reward += reward
    if ep_reward >= 200:
        print("Won! Episode: ", ep+1)
        print("Steps in the episode: ", step)
        break
        
    if (ep+1) % 1000 == 0:
        print("Episode {}. Reward: {}".format(ep+1, ep_reward), end="")    
        print("Value of theta: ", theta, end="")

env.close()


print("*"*100)
print(" Playing the episode with the trained policy")
ep_reward = 0
done = False
state = env.reset()
frames = []

for _ in range(200):
    sleep(0.005)
    action = np.argmax([Q(state,a,theta) for a in actions]) if np.random.random()>10/(ep+1) else np.random.choice(actions)
    env.render()
    new_state, reward, done, _ = env.step(action)
    state = new_state

    ep_reward += reward
print("Reward: ", ep_reward)
env.close()
