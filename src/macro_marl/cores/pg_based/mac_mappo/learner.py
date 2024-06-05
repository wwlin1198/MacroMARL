import torch
import copy
import numpy as np

from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence

from itertools import chain
from .models import Critic

class Learner(object):
    
    def __init__(self, 
                 env, 
                 controller, 
                 memory, 
                 gamma, 
                 ppo_clip_value=0.1, 
                 ppo_epochs=4, 
                 obs_last_action=False,
                 a_lr=1e-2, 
                 c_lr=1e-2,
                 c_mlp_layer_size=[32,32], 
                 c_rnn_layer_size=64,
                 c_train_iteration=1, 
                 c_target_update_freq=50, 
                 tau=0.01,
                 grad_clip_value=None, 
                 grad_clip_norm=None,
                 n_step_TD=0, 
                 TD_lambda=0.0,
                 device='cpu'):

        self.env = env
        self.n_agent = env.n_agent
        self.controller = controller
        self.memory = memory
        self.gamma = gamma

        self.a_lr = a_lr
        self.c_lr = c_lr
        self.c_mlp_layer_size = c_mlp_layer_size
        self.c_rnn_layer_size = c_rnn_layer_size
        self.c_train_iteration = c_train_iteration
        self.c_target_update_freq = c_target_update_freq
        self.ppo_epochs = ppo_epochs
        self.ppo_clip_value = ppo_clip_value
        self.obs_last_action = obs_last_action
        self.tau = tau
        self.grad_clip_value = grad_clip_value
        self.grad_clip_norm = grad_clip_norm

        self.n_step_TD = n_step_TD
        self.TD_lambda = TD_lambda
        self.device = device
        
        self._create_joint_critic()
        self._set_optimizer()


    def train(self, eps, ppo_clip_value, ppo_epochs):

        # Debugging outputs
        # print(f"clip value: {mappo_clip_value}, episodes {ippo_epochs}")
        
        batch, trace_len, epi_len = self.memory.sample()
        batch_size = len(batch)

        ############################# batch for centralized critic ###################################
        cen_batch = self._cat_joint_exps(batch)
        cen_batch, cen_trace_len, cen_epi_len = self._squeeze_cen_exp(cen_batch, 
                                                                      batch_size, 
                                                                      trace_len)

        jobs, reward, n_jobs, terminate, mac_v_b, discount, exp_valid = cen_batch

        if jobs.shape[1] == 0:
            return
        
        Gt = self._get_discounted_return(reward, torch.cat([jobs[:, 0].unsqueeze(1), n_jobs], dim=1), terminate, cen_epi_len, self.joint_critic_tgt_net)
        V_value = self.joint_critic_net(jobs)[0].detach() # prevent gradients from going into critic from actor updates
        delta = Gt - V_value
        adv_values = torch.zeros_like(delta)  # Initialize advantage values tensor  
        # print("adv values", adv_values.shape) 
                # Compute advantages using GAE formula
        for t in reversed(range(len(delta))):
            if t == len(delta) - 1:
                adv_values[t] = delta[t]
            else:
                adv_values[t] = delta[t] + 0.99 * 0.95 * adv_values[t + 1]
        adv_values = (adv_values - adv_values.mean()) / (adv_values.std() + 1e-10)
        assert adv_values.shape[0:2] == mac_v_b.shape[0:2], "Shapes don't match for masking out adv for each agent ..."
        #################### decentralized batch for actor #######################################
 
        dec_batches = self._sep_joint_exps(batch)   
        dec_batches, dec_trace_lens, dec_epi_lens = self._squeeze_dec_exp(dec_batches, 
                                                                          batch_size, 
                                                                          trace_len, 
                                                                          adv_values, 
                                                                          mac_v_b)
        
        
        for agent, batch, trace_len, epi_len in zip(self.controller.agents, 
                                                    dec_batches, 
                                                    dec_trace_lens, 
                                                    dec_epi_lens):

            obs, action, adv, discount, exp_valid = batch

            if obs.shape[1] == 0:
                continue
        
        action_logits = agent.actor_net(obs, eps=eps)[0].detach()
        old_log_pi_a = action_logits.gather(-1, action)


        ##########################################################################################################################
        #PPO updates
        for _ in range(ppo_epochs):

            ##############################  calculate critic loss and optimize the critic_net ####################################
            # NOTE WE SHOULD NOT BACKPROPAGATE CRITIC_NET BY N_STATE
            Gt = self._get_discounted_return(reward,torch.cat([jobs[:,0].unsqueeze(1),
                                            n_jobs],
                                            dim=1), terminate, epi_len, self.joint_critic_tgt_net) 
            
            V_value = self.joint_critic_net(jobs)[0]
            # Clipped value function
            value_clip_margin = 0.2
            V_clipped = V_value + (Gt - V_value).clamp(-value_clip_margin, value_clip_margin)
            
            # Calculate the critic loss using both the clipped and original value estimates
            value_loss_unclipped = F.mse_loss(Gt, V_value)
            value_loss_clipped = F.mse_loss(Gt, V_clipped)
            joint_critic_loss = torch.mean(exp_valid * torch.max(value_loss_unclipped, value_loss_clipped)) / exp_valid.mean()
            
            self.joint_critic_optimizer.zero_grad()
            joint_critic_loss.backward()
            if self.grad_clip_value:
                clip_grad_value_(self.joint_critic_net.parameters(), self.grad_clip_value)
            if self.grad_clip_norm:
                clip_grad_norm_(self.joint_critic_net.parameters(), self.grad_clip_norm)
          
            self.joint_critic_optimizer.step()

            ##############################  calculate actor loss using the updated critic ####################################

            action_logits = agent.actor_net(obs, eps=eps)[0]
            new_log_pi_a = action_logits.gather(-1, action)
            
            ratios = torch.exp(new_log_pi_a - old_log_pi_a)  # Calculating the ratio (pi_theta / pi_theta__old)
            # print("SIZES",ratios.size(),adv.size())
            surr1 = ratios * adv
            surr2 = torch.clamp(ratios, 1 - ppo_clip_value, 1 + ppo_clip_value) * adv
            
            pi_entropy = torch.distributions.Categorical(logits=action_logits).entropy().view(obs.shape[0], 
                                                                                trace_len, 
                                                                                1)
            agent.actor_loss = -torch.mean(exp_valid * torch.min(surr1, surr2))

            agent.actor_optimizer.zero_grad()
            # agent.actor_loss.backward()
            if self.grad_clip_value:
                clip_grad_value_(agent.actor_net.parameters(), self.grad_clip_value)
            if self.grad_clip_norm:
                clip_grad_norm_(agent.actor_net.parameters(), self.grad_clip_norm)
            # agent.actor_optimizer.step()
            total_loss = agent.actor_loss * pi_entropy.mean()
            total_loss.backward()
        
            agent.actor_optimizer.step()


    def update_critic_target_net(self, soft=False):
        if not soft:
            for agent in self.controller.agents:
                agent.critic_tgt_net.load_state_dict(agent.critic_net.state_dict())
        else:
            for agent in self.controller.agents:
                with torch.no_grad():
                    for q, q_targ in zip(agent.critic_net.parameters(), agent.critic_tgt_net.parameters()):
                        q_targ.data.mul_(1 - self.tau)
                        q_targ.data.add_(self.tau * q.data)

    def update_actor_target_net(self, soft=False):
        for agent in self.controller.agents:
            if not soft:
                agent.actor_tgt_net.load_state_dict(agent.actor_net.state_dict())
            else:
                with torch.no_grad():
                    for q, q_targ in zip(agent.actor_net.parameters(), agent.actor_tgt_net.parameters()):
                        q_targ.data.mul_(1 - self.tau)
                        q_targ.data.add_(self.tau * q.data)

    def _set_optimizer(self):
        for agent in self.controller.agents:
            agent.actor_optimizer = Adam(agent.actor_net.parameters(), lr=self.a_lr)
        self.joint_critic_optimizer = Adam(self.joint_critic_net.parameters(), lr=self.c_lr)
        
    def _squeeze_cen_exp(self, cen_batch, batch_size, trace_len):

        """
        squeeze experience for each agent and re-padding
        """

        # seperate elements in the batch
        jobs_b, reward_b, next_jobs_b, terminate_b, mac_valid_b, j_mac_valid_b, exp_valid_b = zip(*cen_batch)
        assert len(jobs_b) == trace_len * batch_size, "number of states mismatch ..."
        assert len(next_jobs_b) == trace_len * batch_size, "number of next states mismatch ..."
        jo_b = torch.cat(jobs_b).view(batch_size, trace_len, -1)
        r_b = torch.cat(reward_b).view(batch_size, trace_len, -1)
        n_jo_b = torch.cat(next_jobs_b).view(batch_size, trace_len, -1)
        t_b = torch.cat(terminate_b).view(batch_size, trace_len, -1)
        mac_v_b = torch.cat(mac_valid_b).view(batch_size, trace_len, -1)
        j_mac_v_b = torch.cat(j_mac_valid_b).view(batch_size, trace_len)
        exp_v_b = torch.cat(exp_valid_b).view(batch_size, trace_len, -1)
        discount_b = torch.pow(torch.ones(jo_b.shape[0],1)*self.gamma, torch.arange(jo_b.shape[1])).unsqueeze(-1) 

        # squeeze process
        squ_epi_len = j_mac_v_b.sum(1)
        squ_jo_b = torch.split_with_sizes(jo_b[j_mac_v_b], list(squ_epi_len))
        squ_r_b = torch.split_with_sizes(r_b[j_mac_v_b], list(squ_epi_len))
        squ_n_jo_b = torch.split_with_sizes(n_jo_b[j_mac_v_b], list(squ_epi_len))
        squ_t_b = torch.split_with_sizes(t_b[j_mac_v_b], list(squ_epi_len))
        squ_mac_v_b = torch.split_with_sizes(mac_v_b[j_mac_v_b], list(squ_epi_len))
        squ_exp_v_b = torch.split_with_sizes(exp_v_b[j_mac_v_b], list(squ_epi_len))
        squ_discount_b = torch.split_with_sizes(discount_b[j_mac_v_b], list(squ_epi_len))

        # re-padding
        squ_jo_b = pad_sequence(squ_jo_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
        squ_r_b = pad_sequence(squ_r_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
        squ_n_jo_b = pad_sequence(squ_n_jo_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
        squ_t_b = pad_sequence(squ_t_b, padding_value=torch.tensor(1.0), batch_first=True).to(self.device)
        squ_mac_v_b = pad_sequence(squ_mac_v_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
        squ_exp_v_b = pad_sequence(squ_exp_v_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
        squ_discount_b = pad_sequence(squ_discount_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)

        squ_cen_batch = (squ_jo_b,
                         squ_r_b,
                         squ_n_jo_b,
                         squ_t_b,
                         squ_mac_v_b,
                         squ_discount_b,
                         squ_exp_v_b)

        return squ_cen_batch, squ_jo_b.shape[1], squ_epi_len
    def _create_joint_critic(self):
        input_dim = self._get_input_shape()
        self.joint_critic_net = Critic(input_dim, 1, self.c_mlp_layer_size, self.c_rnn_layer_size)
        self.joint_critic_tgt_net = Critic(input_dim, 1, self.c_mlp_layer_size, self.c_rnn_layer_size)
        self.joint_critic_tgt_net.load_state_dict(self.joint_critic_net.state_dict())
        
    def _get_input_shape(self):
        if not self.obs_last_action: 
            return sum(self.env.obs_size)
        else:
            return sum([o_dim + a_dim for o_dim, a_dim in zip(*[self.env.obs_size, self.env.n_action])])

    def _sep_joint_exps(self, joint_exps):

        """
        seperate the joint experience for individual agents
        """

        exps = [[] for _ in range(self.n_agent)]
        for o, a_st, avail_a, a, r, j_r, n_o, n_avail_a, t, mac_v, j_mac_v, exp_v in chain(*joint_exps):
            for i in range(self.n_agent):
                exps[i].append([o[i], 
                                a[i], 
                                r[i], 
                                n_o[i], 
                                t, 
                                mac_v[i], 
                                exp_v[i]])
        return exps
    
    def _cat_joint_exps(self, joint_exps):

        """
        seperate the joint experience for individual agents
        """

        exps = []
        for o, a_st, avail_a, a, r, j_r, n_o, n_avail_a, t, mac_v, j_mac_v, exp_v in chain(*joint_exps):
            exps.append([torch.cat(o, dim=1).view(1,-1), 
                         j_r, 
                         torch.cat(n_o, dim=1).view(1,-1), 
                         t, 
                         torch.cat(mac_v).view(1,-1),
                         j_mac_v,
                         exp_v[0]])
        return exps

    def _squeeze_dec_exp(self, dec_batches, batch_size, trace_len, adv_value, j_padded_mac_v_b):

        """
        squeeze experience for each agent and re-padding
        """

        squ_dec_batches = []
        squ_epi_lens = []
        squ_trace_lens = []
        # print(len(dec_batches),batch_size,trace_len,len(adv_value),len(j_padded_mac_v_b))

        for idx, batch in enumerate(dec_batches):
            # seperate elements in the batch
            # print(len(batch))
            obs_b, action_b, reward_b, next_obs_b, terminate_b, mac_valid_b, exp_valid_b = zip(*batch)
            assert len(obs_b) == trace_len * batch_size, "number of states mismatch ..."
            assert len(next_obs_b) == trace_len * batch_size, "number of next states mismatch ..."
            o_b = torch.cat(obs_b).view(batch_size, trace_len, -1)
            a_b = torch.cat(action_b).view(batch_size, trace_len, -1)
            mac_v_b = torch.cat(mac_valid_b).view(batch_size, trace_len)
            exp_v_b = torch.cat(exp_valid_b).view(batch_size, trace_len, -1)
            discount_b = torch.pow(torch.ones(o_b.shape[0],1)*self.gamma, torch.arange(o_b.shape[1])).unsqueeze(-1) 

            # squeeze process
            squ_epi_len = mac_v_b.sum(1)
            assert all(squ_epi_len == j_padded_mac_v_b[:,:,idx].sum(1)), "Valid mask doesn't match ..."
            squ_o_b = torch.split_with_sizes(o_b[mac_v_b], list(squ_epi_len))
            squ_a_b = torch.split_with_sizes(a_b[mac_v_b], list(squ_epi_len))
            squ_adv_b = torch.split_with_sizes(adv_value[j_padded_mac_v_b[:,:,idx]], list(squ_epi_len))
            squ_exp_v_b = torch.split_with_sizes(exp_v_b[mac_v_b], list(squ_epi_len))
            squ_discount_b = torch.split_with_sizes(discount_b[mac_v_b], list(squ_epi_len))

            # re-padding
            squ_o_b = pad_sequence(squ_o_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
            squ_a_b = pad_sequence(squ_a_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
            squ_adv_b = pad_sequence(squ_adv_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
            squ_exp_v_b = pad_sequence(squ_exp_v_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
            squ_discount_b = pad_sequence(squ_discount_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)

            squ_dec_batches.append((squ_o_b,
                                    squ_a_b,
                                    squ_adv_b,
                                    squ_discount_b,
                                    squ_exp_v_b))

            squ_epi_lens.append(squ_epi_len)
            squ_trace_lens.append(squ_o_b.shape[1])

        return squ_dec_batches, squ_trace_lens, squ_epi_lens

    def _get_discounted_return(self, reward, n_state, terminate, epi_len, critic_net):
        Gt = copy.deepcopy(reward)
        for epi_idx, epi_r in enumerate(Gt):
            end_step_idx = epi_len[epi_idx]-1
            if not terminate[epi_idx][end_step_idx]:
                # implement last step bootstrap
                epi_r[end_step_idx] += self.gamma * critic_net(n_state[epi_idx].unsqueeze(0))[0].detach()[:,1:,:][0][end_step_idx]
            for idx in range(end_step_idx-1, -1, -1):
                epi_r[idx] = epi_r[idx] + self.gamma * epi_r[idx+1]
            
            
        return Gt

    def _get_bootstrap_return(self, reward, n_state, terminate, epi_len, critic_net):
        if self.n_step_TD and self.n_step_TD != 1:
            # implement n-step bootstrap
            bootstrap = critic_net(n_state)[0].detach()[:,1:,:]
            Gt = copy.deepcopy(reward)
            for epi_idx, epi_r in enumerate(Gt):
                end_step_idx = epi_len[epi_idx]-1
                if not terminate[epi_idx][end_step_idx]:
                    epi_r[end_step_idx] += self.gamma * bootstrap[epi_idx][end_step_idx]
                for idx in range(end_step_idx-1, -1, -1):
                    if idx > end_step_idx-self.n_step_TD:
                        epi_r[idx] = epi_r[idx] + self.gamma * epi_r[idx+1]
                    else:
                        epi_r[idx] = self._get_n_step_discounted_bootstrap_return(reward[epi_idx][idx:idx+self.n_step_TD], bootstrap[epi_idx][idx+self.n_step_TD-1])
        else:
            Gt = reward + self.gamma * critic_net(n_state)[0].detach()[:,1:,:] * (-terminate + 1)
        return Gt

    def _get_n_step_discounted_bootstrap_return(self, reward, bootstrap):
        discount = torch.pow(torch.ones(1, 1) * self.gamma, torch.arange(self.n_step_TD)).view(self.n_step_TD, 1)
        Gt = torch.sum(discount * reward) + self.gamma**self.n_step_TD * bootstrap
        return Gt

    def _get_td_lambda_return(self, batch_size, trace_len, epi_len, reward, n_state, terminate, critic_net):
        # calculate MC returns
        Gt = self._get_discounted_return(reward, n_state, terminate, epi_len, critic_net)
        # calculate n-step bootstrap returns
        self.n_step_TD = 0
        n_step_part = self._get_bootstrap_return(reward, n_state, terminate, epi_len, critic_net)
        for n in range(2, trace_len):
            self.n_step_TD=n
            next_n_step_part = self._get_bootstrap_return(reward, n_state, terminate, epi_len, critic_net)
            n_step_part = torch.cat([n_step_part, next_n_step_part], dim=-1)
        # calculate the lmda for n-step bootstrap part
        lmdas = torch.pow(torch.ones(1,1)*self.TD_lambda, torch.arange(trace_len-1)).repeat(trace_len, 1).unsqueeze(0).repeat(batch_size,1,1)
        mask = (torch.arange(trace_len).view(-1,1) + torch.arange(trace_len-1).view(1,-1)).squeeze(0).repeat(batch_size,1,1)
        mask = mask >= epi_len.view(batch_size, -1, 1)-1
        lmdas[mask] = 0.0
        # calculate the lmda for MC part
        MC_lmdas = torch.zeros_like(Gt)
        for epi_id, length in enumerate(epi_len):
            last_step_lmda = torch.pow(torch.ones(1,1)*self.TD_lambda, torch.arange(length-1,-1,-1)).view(-1,1)
            MC_lmdas[epi_id][0:length] += last_step_lmda
        # TD LAMBDA RETURN
        Gt = (1 - self.TD_lambda) * torch.sum(lmdas * n_step_part, dim=-1, keepdim=True) +  MC_lmdas * Gt
        return Gt
