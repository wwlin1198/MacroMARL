import torch
import copy
import numpy as np

from torch.optim import Adam
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence

from itertools import chain

class Learner(object):
    
    def __init__(self, 
                 env, 
                 controller, 
                 memory, 
                 gamma, 
                 obs_last_action=False,
                 a_lr=1e-2, 
                 c_lr=1e-2, 
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
        self.c_train_iteration = c_train_iteration
        self.c_target_update_freq = c_target_update_freq
        self.obs_last_action = obs_last_action
        self.tau = tau
        self.grad_clip_value = grad_clip_value
        self.grad_clip_norm = grad_clip_norm
        self.n_step_TD = n_step_TD
        self.TD_lambda = TD_lambda
        self.device = device

        self._set_optimizer()

    def train(self, eps, c_hys_value, adv_hys_value, etrpy_w, critic_hys=False, adv_hys=False):

        batch, trace_len, epi_len = self.memory.sample()
        batch_size = len(batch)

        dec_batches = self._sep_joint_exps(batch)
        dec_batches, trace_lens, epi_lens = self._squeeze_dec_exp(dec_batches, 
                                                                  batch_size, 
                                                                  trace_len)

        for agent, batch, trace_len, epi_len in zip(self.controller.agents, 
                                                    dec_batches, 
                                                    trace_lens, 
                                                    epi_lens):

            state, obs, action, reward, n_state, terminate, discount, exp_valid = batch 

            if obs.shape[1] == 0:
                continue

            ##############################  calculate critic loss and optimize the critic_net ####################################
            for _ in range(self.c_train_iteration):
                if not self.TD_lambda:
                    # NOTE WE SHOULD NOT BACKPROPAGATE CRITIC_NET BY N_STATE
                    Gt = self._get_bootstrap_return(reward,
                                                    n_state,
                                                    discount,
                                                    terminate, 
                                                    epi_len, 
                                                    agent.critic_tgt_net)
                else:
                    Gt = self._get_td_lambda_return(obs.shape[0], 
                                                    trace_len, 
                                                    epi_len, 
                                                    reward, 
                                                    n_state,
                                                    terminate, 
                                                    agent.critic_tgt_net)

                TD = Gt - agent.critic_net(state)
                if critic_hys:
                    TD = torch.max(TD*c_hys_value, TD)
                    
                # v_val clipping instead 
                V_value = agent.critic_net(state)
                value_clip_margin = eps - 0.0001
                V_clipped = V_value + (Gt - V_value).clamp(-value_clip_margin, value_clip_margin)
                value_loss_unclipped = (Gt - V_value) ** 2
                value_loss_clipped = (Gt - V_clipped) ** 2

                agent.critic_loss = torch.mean(exp_valid * torch.max(value_loss_unclipped, value_loss_clipped))


                # agent.critic_loss = torch.sum(exp_valid * TD * TD) / torch.sum(exp_valid)

                agent.critic_optimizer.zero_grad()
                agent.critic_loss.backward()
                if self.grad_clip_value:
                    clip_grad_value_(agent.critic_net.parameters(), self.grad_clip_value)
                if self.grad_clip_norm:
                    clip_grad_norm_(agent.critic_net.parameters(), self.grad_clip_norm)
                agent.critic_optimizer.step()

            ##############################  calculate actor loss using the updated critic ####################################
            V_value = agent.critic_net(state).detach()

            adv_value = Gt - V_value
            if adv_hys:
                adv_value = torch.max(adv_value*adv_hys_value, adv_value)

            action_logits = agent.actor_net(obs, eps=eps)[0]
            log_pi_a = action_logits.gather(-1, action)
            pi_entropy = torch.distributions.Categorical(logits=action_logits).entropy().view(obs.shape[0], 
                                                                                              trace_len, 
                                                                                              1)
            actor_loss = torch.sum(exp_valid * discount * (log_pi_a * adv_value + etrpy_w * pi_entropy), dim=1)
            agent.actor_loss = -1 * torch.sum(actor_loss) / exp_valid.sum()

            agent.actor_optimizer.zero_grad()
            agent.actor_loss.backward()
            if self.grad_clip_value:
                clip_grad_value_(agent.actor_net.parameters(), self.grad_clip_value)
            if self.grad_clip_norm:
                clip_grad_norm_(agent.actor_net.parameters(), self.grad_clip_norm)
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
            agent.critic_optimizer = Adam(agent.critic_net.parameters(), lr=self.c_lr)

    def _sep_joint_exps(self, joint_exps):

        """
        seperate the joint experience for individual agents
        """

        exps = [[] for _ in range(self.n_agent)]
        for s, o, a_st, avail_a, a, r, j_r, n_s, n_o, n_avail_a, t, mac_v, j_mac_v, exp_v in chain(*joint_exps):
            for i in range(self.n_agent):
                exps[i].append([s,
                                o[i], 
                                a_st[i],
                                a[i], 
                                r[i], 
                                n_s,
                                t, 
                                mac_v[i], 
                                exp_v[i]])
        return exps

    def _squeeze_dec_exp(self, dec_batches, batch_size, trace_len):

        """
        squeeze experience for each agent and re-padding
        """

        squ_dec_batches = []
        squ_epi_lens = []
        squ_trace_lens = []

        for idx, batch in enumerate(dec_batches):
            # seperate elements in the batch
            (state_b, 
             obs_b, 
             action_start_b, 
             action_b, 
             reward_b, 
             next_state_b, 
             terminate_b, 
             mac_valid_b, 
             exp_valid_b) = zip(*batch)

            assert len(state_b) == trace_len * batch_size, "number of states mismatch ..."
            assert len(next_state_b) == trace_len * batch_size, "number of next states mismatch ..."

            s_b = torch.cat(state_b).view(batch_size, trace_len, -1)
            o_b = torch.cat(obs_b).view(batch_size, trace_len, -1)
            a_st_b = torch.cat(action_start_b).view(batch_size, trace_len)
            a_b = torch.cat(action_b).view(batch_size, trace_len, -1)
            r_b = torch.cat(reward_b).view(batch_size, trace_len, -1)
            n_s_b = torch.cat(next_state_b).view(batch_size, trace_len, -1)
            t_b = torch.cat(terminate_b).view(batch_size, trace_len, -1)
            mac_v_b = torch.cat(mac_valid_b).view(batch_size, trace_len)
            exp_v_b = torch.cat(exp_valid_b).view(batch_size, trace_len, -1)
            discount_b = torch.pow(torch.ones(s_b.shape[0],1)*self.gamma, torch.arange(s_b.shape[1])).unsqueeze(-1) 

            if not all(a_st_b.sum(1) == mac_v_b.sum(1)):
                mask = a_st_b.sum(1) != mac_v_b.sum(1)
                selected_items = a_st_b[mask]
                indices = torch.cat([i[-1].view(-1,2) for i in torch.split_with_sizes(selected_items.nonzero(as_tuple=False), 
                                                                                      list(selected_items.sum(1)))], 
                                                                                      dim=0)
                selected_items.scatter_(-1, indices[:,1].view(-1,1), 0.0)
                a_st_b[mask] = selected_items
            assert all(a_st_b.sum(1) == mac_v_b.sum(1)), "mask for mac strat does not match with mask of mac done ..."

            # squeeze process
            squ_epi_len = mac_v_b.sum(1)
            squ_s_b = torch.split_with_sizes(s_b[a_st_b], list(squ_epi_len))
            squ_o_b = torch.split_with_sizes(o_b[mac_v_b], list(squ_epi_len))
            squ_a_b = torch.split_with_sizes(a_b[mac_v_b], list(squ_epi_len))
            squ_r_b = torch.split_with_sizes(r_b[mac_v_b], list(squ_epi_len))
            squ_n_s_b = torch.split_with_sizes(n_s_b[mac_v_b], list(squ_epi_len))
            squ_t_b = torch.split_with_sizes(t_b[mac_v_b], list(squ_epi_len))
            squ_exp_v_b = torch.split_with_sizes(exp_v_b[mac_v_b], list(squ_epi_len))
            squ_discount_b = torch.split_with_sizes(discount_b[mac_v_b], list(squ_epi_len))

            # re-padding
            squ_s_b = pad_sequence(squ_s_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
            squ_o_b = pad_sequence(squ_o_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
            squ_a_b = pad_sequence(squ_a_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
            squ_r_b = pad_sequence(squ_r_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
            squ_n_s_b = pad_sequence(squ_n_s_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
            squ_t_b = pad_sequence(squ_t_b, padding_value=torch.tensor(1.0), batch_first=True).to(self.device)
            squ_exp_v_b = pad_sequence(squ_exp_v_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
            squ_discount_b = pad_sequence(squ_discount_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)

            squ_dec_batches.append((squ_s_b,
                                    squ_o_b,
                                    squ_a_b,
                                    squ_r_b,
                                    squ_n_s_b,
                                    squ_t_b,
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
                epi_r[end_step_idx] += self.gamma * critic_net(n_state[epi_idx].unsqueeze(0)).detach()[0][end_step_idx]
            for idx in range(end_step_idx-1, -1, -1):
                epi_r[idx] = epi_r[idx] + self.gamma * epi_r[idx+1]
        return Gt

    def _get_bootstrap_return(self, reward, n_state, discount, terminate, epi_len, critic_net):
        mac_discount = discount / torch.cat((self.gamma**-1*torch.ones((discount.shape[0],1,1)),
                                             discount[:,0:-1,:]),
                                             axis=1) 
        mask = mac_discount.isnan()
        mac_discount[mask] = 0.0
        if self.n_step_TD and self.n_step_TD != 1:
            # implement n-step bootstrap
            bootstrap = critic_net(n_state).detach()
            Gt = copy.deepcopy(reward)
            for epi_idx, epi_r in enumerate(Gt):
                end_step_idx = epi_len[epi_idx]-1
                if not terminate[epi_idx][end_step_idx]:
                    epi_r[end_step_idx] += mac_discount[epi_idx][end_step_idx] * bootstrap[epi_idx][end_step_idx]
                for idx in range(end_step_idx-1, -1, -1):
                    if idx > end_step_idx-self.n_step_TD:
                        epi_r[idx] = epi_r[idx] + mac_discount[epi_idx][idx] * epi_r[idx+1]
                    else:
                        if idx == 0:
                            epi_r[idx] = self._get_n_step_discounted_bootstrap_return(reward[epi_idx][idx:idx+self.n_step_TD], 
                                                                                      bootstrap[epi_idx][idx+self.n_step_TD-1],
                                                                                      discount[epi_idx][idx:idx+self.n_step_TD] / self.gamma**-1)
                        else:
                            epi_r[idx] = self._get_n_step_discounted_bootstrap_return(reward[epi_idx][idx:idx+self.n_step_TD], 
                                                                                      bootstrap[epi_idx][idx+self.n_step_TD-1],
                                                                                      discount[epi_idx][idx:idx+self.n_step_TD] / discount[epi_idx][idx-1])
        else:
            Gt = reward + mac_discount * critic_net(n_state).detach() * (-terminate + 1)
        return Gt

    def _get_n_step_discounted_bootstrap_return(self, reward, bootstrap, discount):
        rewards = torch.cat((reward, bootstrap.reshape(-1,1)), axis=0)
        discounts = torch.cat((torch.ones((1,1)), discount), axis=0)
        Gt = torch.sum(discounts * rewards) 
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
