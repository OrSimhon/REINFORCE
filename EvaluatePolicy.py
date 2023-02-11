import torch


def log_summary(ep_ret, ep_num):
    ep_ret = str(round(ep_ret, 2))  # Round decimal places for more aesthetic logging messages

    # Print logging statements
    print(flush=True)
    print("------------------------- Episode #{} -------------------------".format(ep_num), flush=True)
    print("Episodic Return: {}".format(ep_ret), flush=True)
    print("---------------------------------------------------------------")
    print(flush=True)


def rollout(policy, env, render):
    while True:  # Rollout until user kills process
        obs = env.reset()
        done = False
        t = 0 # number of time steps so far
        ep_ret = 0  # Episodic return

        while not done:
            t += 1
            if render: # Render environment if specified, off by default
                env.render()

            action = torch.argmax(policy(obs).detach()).item() # Query deterministic action from policy and run it
            obs, rew, done, _ = env.step(action)
            ep_ret += rew # Sum all episodic rewards as we go along

        # returns episodic length and return in this iteration
        yield ep_ret


def eval_policy(policy, env, render=False):
    # Rollout with the policy and environment, and log each episode's data
    for ep_num, ep_ret in enumerate(rollout(policy, env, render)):
        log_summary(ep_ret=ep_ret, ep_num=ep_num)
