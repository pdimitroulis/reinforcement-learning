import gym
import numpy as np
import matplotlib.pyplot as plt
import sys

# NOTE:
# - With initial exploration, 1st achievement at the ~450th episode.
# - Without it, 1st achievement at the ~1000th episode.

env = gym.make("MountainCar-v0")

# metrics
LEARNING_RATE = 0.1
DISCOUNT = 0.95     # weight.  Shows how important is future reward over current reward.
EPISODES = 4000     # Each episode lasts a specific time. It's actually how many trials it will attempt.
SHOW_EVERY = 2000   
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# --- randomness/exploration metrics ---
epsilon = 0.5 # range [0,1] . How much randomness in its actions.
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2    # The episode that stops epsilon decaying
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# rewards for each episode
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

# discretize states
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

for episode in range(EPISODES):
    episode_reward = 0
    if episode % SHOW_EVERY == 0:       #every several episodes show environment, just to know it's alive and running.
        print(episode)
        render = True
    else:
        render = False
    discrete_state = get_discrete_state(env.reset())     # get initial state

    done = False
    while not done:
        # --- uncomment your option ---
        # 1) Initial exploration
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        # 2) Without initial exploration
        # action = np.argmax(q_table[discrete_state])

        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            # new_q  formula
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state+(action,)] = new_q
        elif new_state[0] >= env.goal_position:
            print(f"We made it on episode {episode}")
            sys.stdout.flush()
            q_table[discrete_state + (action,)] = 0     # perfect reward for achieving the goal.

        discrete_state = new_discrete_state
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)

    # save your model (in order to pick the best, based on the graph)
    if not episode % 10:
        np.save(f"qtables/{episode}-qtable.npy", q_table)

    # collect reward data
    if not episode % SHOW_EVERY:  # every episode
        average_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])   # avg of the last SHOW_EVERY episodes.
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

        print(f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max: {max(ep_rewards[-SHOW_EVERY:])}")
env.close()

# plot reward data
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="avg")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max")
plt.legend(loc=1)
plt.show()
