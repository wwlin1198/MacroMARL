import time
import numpy as np

from macro_marl.cores.pg_based.mac_ippov2.memory import Memory_epi, Memory_rand
from macro_marl.cores.pg_based.mac_ippov2.controller import MAC
from macro_marl.cores.pg_based.mac_ippov2.envs_runner import EnvsRunner
from macro_marl.cores.pg_based.mac_ippov2.learner import Learner
from macro_marl.cores.pg_based.mac_ippov2.utils import Linear_Decay, save_train_data, save_test_data, save_checkpoint, load_checkpoint, save_policies

import wandb

class MacIPPOV2(object):

    def __init__(self,
            env,
            env_terminate_step, 
            n_env, 
            n_agent, 
            total_epi, 
            gamma, 
            ppo_clip_value,
            ppo_epochs,
            a_lr, 
            c_lr, 
            eps_start, 
            eps_end, 
            eps_stable_at, 
            train_freq, 
            c_target_update_freq, 
            c_target_soft_update, 
            tau, 
            n_step_TD, 
            TD_lambda, 
            a_mlp_layer_size, 
            a_rnn_layer_size, 
            c_mlp_layer_size, 
            c_rnn_layer_size, 
            grad_clip_value, 
            grad_clip_norm, 
            obs_last_action, 
            eval_policy, 
            eval_freq, 
            eval_num_epi, 
            sample_epi, 
            trace_len, 
            seed, 
            run_id, 
            save_dir, 
            resume, 
            device, 
            *args, 
            **kwargs):

        self.total_epi = total_epi
        self.train_freq = train_freq
        self.eval_policy = eval_policy
        self.eval_freq = eval_freq
        self.eval_num_epi = eval_num_epi
        self.ppo_clip_value = ppo_clip_value
        self.ppo_epochs = ppo_epochs
        self.sample_epi = sample_epi
        self.c_target_update_freq = c_target_update_freq
        self.c_target_soft_update = c_target_soft_update
        self.run_id = run_id
        self.save_dir = save_dir
        self.resume = resume


        # collect params
        actor_params = {'a_mlp_layer_size': a_mlp_layer_size,
                        'a_rnn_layer_size': a_rnn_layer_size}

        critic_params = {'c_mlp_layer_size': c_mlp_layer_size,
                         'c_rnn_layer_size': c_rnn_layer_size}

        hyper_params = {'a_lr': a_lr,
                        'c_lr': c_lr,
                        'ppo_clip_value': ppo_clip_value,
                        'ppo_epochs': ppo_epochs,
                        'tau': tau,
                        'grad_clip_value': grad_clip_value,
                        'grad_clip_norm': grad_clip_norm,
                        'n_step_TD': n_step_TD,
                        'TD_lambda': TD_lambda,
                        'device': device}

        self.env = env
        # create buffer
        if sample_epi:
            self.memory = Memory_epi(env.obs_size, env.n_action, obs_last_action, size=train_freq)
        else:
            self.memory = Memory_rand(trace_len, env.obs_size, env.n_action, obs_last_action, size=train_freq)
        # cretate controller
        self.controller = MAC(self.env, obs_last_action, **actor_params, **critic_params, device=device) 
        # create parallel envs runner
        self.envs_runner = EnvsRunner(self.env, n_env, self.controller, self.memory, env_terminate_step, gamma, seed, obs_last_action)
        # create learner
        self.learner = Learner(self.env, self.controller, self.memory, gamma, **hyper_params)
        # create epsilon calculator for implementing e-greedy exploration policy
        self.eps_call = Linear_Decay(eps_stable_at, eps_start, eps_end)
        # create entropy loss weight calculator
        # record evaluation return
        self.eval_returns = []

        wandb.login(key='1953b06a2318828bc531085d9e76a250f82840fd')
        wandb.init(project=f'MAC_IPPOV2_warehouseE_{ppo_clip_value}_{ppo_epochs}_v1')


    def learn(self):
        epi_count = 0
        if self.resume:
            epi_count, self.eval_returns = load_checkpoint(self.run_id, self.save_dir, self.controller, self.envs_runner)

        while epi_count < self.total_epi:

            if self.eval_policy and epi_count % (self.eval_freq - (self.eval_freq % self.train_freq)) == 0:
                self.envs_runner.run(n_epis=self.eval_num_epi, test_mode=True)
                assert len(self.envs_runner.eval_returns) >= self.eval_num_epi, "Did Not Evaluate Sufficient Episodes ..."
                self.eval_returns.append(np.mean(self.envs_runner.eval_returns[-self.eval_num_epi:]))
                self.envs_runner.eval_returns = []
                print(f"{[self.run_id]} Finished: {epi_count}/{self.total_epi} Evaluate learned policies with averaged returns {self.eval_returns[-1]} ...", flush=True)
                
                wandb.log({'mean_return':self.eval_returns[-1],'episodes': epi_count})

                # save the best policy
                if self.eval_returns[-1] == np.max(self.eval_returns):
                    save_policies(self.run_id, self.controller.agents, self.save_dir)


            # update eps
            eps = self.eps_call.get_value(epi_count)

            # let envs run a certain number of episodes according to train_freq
            self.envs_runner.run(eps=eps, n_epis=self.train_freq)

            # perform ppo update
            self.learner.train(eps, self.ppo_clip_value, self.ppo_epochs)
            if not self.sample_epi:
                self.memory.buf.clear()

            epi_count += self.train_freq

            # update target net
            if self.c_target_soft_update:
                self.learner.update_critic_target_net(soft=True)
                self.learner.update_actor_target_net(soft=True)
            elif epi_count % self.c_target_update_freq == 0:
                self.learner.update_critic_target_net()
                self.learner.update_actor_target_net()

        ################################ saving in the end ###################################
        save_train_data(self.run_id, self.envs_runner.train_returns, self.save_dir)
        save_test_data(self.run_id, self.eval_returns, self.save_dir)
        save_checkpoint(self.run_id, 
                        epi_count, 
                        self.eval_returns, 
                        self.controller, 
                        self.envs_runner, 
                        self.save_dir)
        self.envs_runner.close()

        print(f"{[self.run_id]} Finish Training ... ", flush=True)
