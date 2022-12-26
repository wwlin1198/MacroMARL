import pickle
import torch
import os
import random
import numpy as np

class Agent:

    def __init__(self):
        self.idx = None
        self.actor_net = None
        self.actor_optimizer = None
        self.actor_loss = None

class Linear_Decay(object):

    def __init__ (self, total_steps, init_value, end_value):
        self.total_steps = total_steps
        self.init_value = init_value
        self.end_value = end_value

    def get_value(self, step):
        frac = min(float(step) / self.total_steps, 1.0)
        return self.init_value + frac * (self.end_value-self.init_value)

def save_policies(run_id, agents, save_dir):
    for agent in agents:
        PATH = "./policy_nns/" + save_dir + "/" + str(run_id) + "_agent_" + str(agent.idx) + ".pt"
        torch.save(agent.actor_net, PATH)

def save_train_data(run_id, data, save_dir):
    with open("./performance/" + save_dir + "/train/train_perform" + str(run_id) + ".pickle", 'wb') as handle:
        pickle.dump(data, handle)

def save_test_data(run_id, data, save_dir):
    with open("./performance/" + save_dir + "/test/test_perform" + str(run_id) + ".pickle", 'wb') as handle:
        pickle.dump(data, handle)

def save_checkpoint(run_id, epi_count, eval_returns, controller, learner, envs_runner, save_dir, max_save=2):

    PATH = "./performance/" + save_dir + "/ckpt/" + str(run_id) + "_genric_" + "{}.tar"

    for n in list(range(max_save-1, 0, -1)):
        os.system('cp -rf ' + PATH.format(n) + ' ' + PATH.format(n+1) )
    PATH = PATH.format(1)

    torch.save({
                'epi_count': epi_count,
                'random_state': random.getstate(),
                'np_random_state': np.random.get_state(),
                'torch_random_state': torch.random.get_rng_state(),
                'envs_runner_returns': envs_runner.train_returns,
                'eval_returns': eval_returns,
                'joint_critic_net_state_dict': learner.joint_critic_net.state_dict(),
                'joint_critic_tgt_net_state_dict': learner.joint_critic_tgt_net.state_dict(),
                'joint_critic_optimizer_state_dict': learner.joint_critic_optimizer.state_dict()
                }, PATH)

    for idx, parent in enumerate(envs_runner.parents):
        parent.send(('get_rand_states', None))
    for idx, parent in enumerate(envs_runner.parents):
        PATH = "./performance/" + save_dir + "/ckpt/" + str(run_id) + "_env_rand_states_" + str(idx) + "{}.tar"
        rand_states = parent.recv()
        for n in list(range(max_save-1, 0, -1)):
            os.system('cp -rf ' + PATH.format(n) + ' ' + PATH.format(n+1) )
        PATH = PATH.format(1)
        torch.save(rand_states, PATH)

    for idx, agent in enumerate(controller.agents):
        PATH = "./performance/" + save_dir + "/ckpt/" + str(run_id) + "_agent_" + str(idx) + "{}.tar"

        for n in list(range(max_save-1, 0, -1)):
            os.system('cp -rf ' + PATH.format(n) + ' ' + PATH.format(n+1) )
        PATH = PATH.format(1)

        torch.save({
                    'actor_net_state_dict': agent.actor_net.state_dict(),
                    'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                    },PATH)

def load_checkpoint(run_id, save_dir, controller, learner, envs_runner):

    # load generic stuff
    PATH = "./performance/" + save_dir + "/ckpt/" + str(run_id) + "_genric_" + "1.tar"
    ckpt = torch.load(PATH)
    epi_count = ckpt['epi_count']
    random.setstate(ckpt['random_state'])
    np.random.set_state(ckpt['np_random_state'])
    torch.set_rng_state(ckpt['torch_random_state'])
    envs_runner.train_returns = ckpt['envs_runner_returns']
    eval_returns = ckpt['eval_returns']
    learner.joint_critic_net.load_state_dict(ckpt['joint_critic_net_state_dict'])
    learner.joint_critic_tgt_net.load_state_dict(ckpt['joint_critic_tgt_net_state_dict'])
    learner.joint_critic_optimizer.load_state_dict(ckpt['joint_critic_optimizer_state_dict'])

    # load random states in all workers
    for idx, parent in enumerate(envs_runner.parents):
        PATH = "./performance/" + save_dir + "/ckpt/" + str(run_id) + "_env_rand_states_" + str(idx) + "1.tar"
        rand_states = torch.load(PATH)
        parent.send(('load_rand_states', rand_states))

    # load actor and ciritc models
    for idx, agent in enumerate(controller.agents):
        PATH = "./performance/" + save_dir + "/ckpt/" + str(run_id) + "_agent_" + str(idx) + "1.tar"
        ckpt = torch.load(PATH)
        agent.actor_net.load_state_dict(ckpt['actor_net_state_dict'])
        agent.actor_optimizer.load_state_dict(ckpt['actor_optimizer_state_dict'])

    return epi_count, eval_returns
