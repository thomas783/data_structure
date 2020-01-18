from Game import Game

import win32api
import win32con
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

import time
import pdb

H_2 = 200
H_1 = 300
D = 80 * 69

gamma = 0.99  # discount factor
learning_rate = 1e-3
batch_size = 5
resume = False
model_name = 'qwop.torch.model'
 

class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.input = nn.Linear(D, H_1)
        self.hidden = nn.Linear(H_1, H_2)
        self.hidden_2 = nn.Linear(H_2, 7)

    def forward(self, x):
        x = nn.functional.leaky_relu(self.input(x))
        x = nn.functional.leaky_relu(self.hidden(x))
        x = nn.functional.softmax(self.hidden_2(x), dim=0)
        return x

def map_action(action):
    return (['a','s','q','w','o','p'])[action]
  

def discount_rewards(reward_log):
    #reward_log = reward_log.ravel()
    discount = 0
    shape = np.shape(reward_log)
    discounted_rewards = np.zeros(shape)
    for idx in reversed(range(0, discounted_rewards.size )):
        if reward_log[idx] != 0: discount = 0 
        discount = gamma * discount + reward_log[idx]
        discounted_rewards[idx] = discount
    return discounted_rewards


def main():
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    if resume: policy.load_state_dict(torch.load(model_name))

    env = Game()
    env.start()

    observation, reward, done = env.execute_action('n',0)

    prev_x = None
    curr_x = observation
    reward_pool = []
    prob_pool = []
    steps = 0
    reward_sum = 0
    game = 0
    last_reward = 0
    running_reward = 0
    in_game_step = 0

    while True:

        for step in range(50):

            in_game_step += 1

            x_diff = curr_x - prev_x if prev_x is not None else curr_x
            prev_x = curr_x
            x = Variable(torch.from_numpy(x_diff).float())

            forward = policy(x)
            forward_1 = forward[:6] 
            forward_2 = forward[6]
            out_dist = Categorical(forward_1)
            action = out_dist.sample()
            forwardtime = float(forward_2)
            if forwardtime<0.1:
                forwardtime = 0.1
            elif forwardtime>1 :
                forwardtime = 1.0
            else :
                forwardtime = forwardtime
            curr_x, reward, done = env.execute_action(map_action(action.item()),forwardtime)
            # print('time:', forwardtime)

            if done:
                in_game_step = 0

            if in_game_step > 0 and in_game_step % 100 == 0:
                print('should have reloaded')
                reward = env.get_score()[1]
                env.reload()

            prob_pool.append(out_dist.log_prob(action))
            prob_pool.append(forward_2)

            reward_pool.append(reward)
            reward_sum += reward

            if reward != last_reward and reward != 0:
                game += 1
                print(f"{time.time()} episode: {steps}, game: {game} reward {reward}")
                last_reward = reward

        steps += 1
        print(f'End of sub batch {steps}')

        if steps > 0 and steps % batch_size == 0:
            print(f'calling backprop batch {steps}')
            discounted_rewards = np.array(discount_rewards(reward_pool)).ravel()
            discounted_rewards -= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards)

            optimizer.zero_grad()
            policy_loss = []

            #pdb.set_trace()

            for prob, dis_reward in zip(prob_pool, discounted_rewards):
                policy_loss.append(-prob * dis_reward)

            loss_fn = torch.stack(policy_loss).sum()
            loss_fn.backward()
            optimizer.step()

            del prob_pool[:]
            del reward_pool[:]

            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print(f'Batch reward total was ${reward_sum} running mean: #{running_reward}')
            torch.save(policy.state_dict(), model_name)
            reward_sum = 0
            prev_x = None
            curr_x, reward, done = env.reload()
            last_reward = 0 

if __name__ == '__main__':
    win32api.keybd_event(0xA4, 0, 0x00, 0)
    win32api.keybd_event(0x09, 0, 0x00, 0)
    win32api.keybd_event(0xA4, 0, 0x02, 0)
    win32api.keybd_event(0x09, 0, 0x02, 0)
    time.sleep(1)
    win32api.keybd_event(0x20, 0, 0x00, 0)
    win32api.keybd_event(0x20, 0, 0x02, 0)
    time.sleep(2.4)
    main()