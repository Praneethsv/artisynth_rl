# for mu, sigma, h in zip(mean, std, horizon):
#     normal = Normal(mu, sigma)
#     action_cat, mean_cat, log_prob_cat = [], [], []
#     for i in range(h):  #
#         print('h: ', h)
#         x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
#         y_t = torch.tanh(x_t)  # muscle activations
#         # print('y_t', y_t)
#         action = y_t * self.action_scale + self.action_bias
#         print('action: ', action)
#         action_cat.append(action)
#         log_prob = normal.log_prob(x_t)
#         # Enforcing Action Bound
#         log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
#         log_prob = log_prob.sum(1, keepdim=True)
#         log_prob_cat.append(log_prob)
#         mean = torch.tanh(mean) * self.action_scale + self.action_bias
#         # print('log_prob : ', log_prob)
#         mean_cat.append(mean)
#     action_cat = torch.cat(action_cat).flatten()
#     # print('actions_cat size: ', action_cat.shape)
#     action_cat = torch.cat([action_cat, torch.zeros(202 - action_cat.shape[0]).to('cuda')])
#     # print('actions after padding: ', action_cat.shape)
#     mean_cat = torch.cat(mean_cat).flatten()
#     mean_cat = torch.cat([mean_cat, torch.zeros(202 - mean_cat.shape[0]).to('cuda')])
#     log_prob_cat = torch.cat(log_prob_cat).flatten()
#     # print('actions before stacking: ', action_cat.shape)
#     actions_stack.append(action_cat.flatten())
#     means_stack.append(mean_cat.flatten())
#     log_probs_stack.append(log_prob_cat)
#     actions_stack = torch.stack(actions_stack)
#     means_stack = torch.stack(means_stack)
#     log_probs_stack = torch.stack(log_probs_stack)