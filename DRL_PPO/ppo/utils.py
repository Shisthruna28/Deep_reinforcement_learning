import math
import torch


def calc_logprob(mu, logstd, actions):
    p1 = - ((mu - actions) ** 2) / (2 * torch.exp(logstd).clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd)))
    return p1 + p2

def calc_adv_ref(trajectory, net_crt, states_v, device="cpu",gamma=0.99,gae_lambda=0.95):

    values_v = net_crt(states_v)
    values = values_v.squeeze().data.cpu().numpy()
    last_gae = 0.0
    result_adv = []
    result_ref = []
    for val, next_val, (exp,) in zip(reversed(values[:-1]), reversed(values[1:]),
                                     reversed(trajectory[:-1])):
        if exp.done:
            delta = exp.reward - val
            last_gae = delta
        else:
            delta = exp.reward + gamma * next_val - val
            last_gae = delta + gamma * gae_lambda * last_gae
        result_adv.append(last_gae)
        result_ref.append(last_gae + val)

    adv_v = torch.FloatTensor(list(reversed(result_adv))).to(device)
    ref_v = torch.FloatTensor(list(reversed(result_ref))).to(device)
    return adv_v, ref_v

