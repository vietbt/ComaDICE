import numpy as np


def load_smac_data(data, n_agents):
    states = data["states"]
    obs = data["obs"]
    avails = data["avails"]
    actions = data["actions"]
    rewards = data["rewards"]
    dones = data["dones"]
    actives = data["actives"]

    if len(states.shape) == 3:
        data["states"] = states = np.repeat(states[:, :, None, :], n_agents, axis=2)
    if len(rewards.shape) == 2:
        data["rewards"] = rewards = np.repeat(rewards[:, :, None], n_agents, axis=2)
    if len(dones.shape) == 2:
        data["dones"] = dones = np.repeat(dones[:, :, None], n_agents, axis=2)
    if len(actives.shape) == 2:
        data["actives"] = actives = np.repeat(actives[:, :, None], n_agents, axis=2)

    rewards = np.cumsum(rewards[:, ::-1], 1)[:, ::-1]
    _done_ids = np.cumsum(actives.sum(1), 0)
    
    _states = states[:, :-1][actives]
    _obs = obs[:, :-1][actives]
    _avails = avails[:, :-1][actives]
    _actions = actions[actives]
    _rewards = rewards[actives]
    _dones = dones[actives]

    _next_states = states[:, 1:][actives]
    _next_obs = obs[:, 1:][actives]
    _is_inits = np.zeros_like(rewards)
    _is_inits[:, 0] = 1
    _is_inits = _is_inits[actives]

    _states = _states.reshape(-1, n_agents, _states.shape[-1])
    _obs = _obs.reshape(-1, n_agents, _obs.shape[-1])
    _avails = _avails.reshape(-1, n_agents, _avails.shape[-1])
    _actions = _actions.reshape(-1, n_agents, 1)
    _rewards = _rewards.reshape(-1, n_agents)
    _dones = _dones.reshape(-1, n_agents)

    _next_states = _next_states.reshape(-1, n_agents, _next_states.shape[-1])
    _next_obs = _next_obs.reshape(-1, n_agents, _next_obs.shape[-1])
    _is_inits = _is_inits.reshape(-1, n_agents)
    
    return _states, _obs, _actions, _done_ids, _rewards, _next_states, _next_obs, _avails, _is_inits


def load_mamujoco_data(data):
    states = data["states"]
    obs = data["obs"]
    actions = data["actions"]
    rewards = data["rewards"]
    dones = data["dones"]
    actives = data["actives"]

    is_inits = np.zeros_like(rewards)
    is_inits[:, 0] = 1
    is_inits = is_inits[actives]

    _states = states[:, :-1][actives]
    _obs = obs[:, :-1][actives]
    _actions = actions[:, :-1][actives]
    _rewards = rewards[actives].reshape(-1, 1)
    _masks = 1 - dones[actives].reshape(-1, 1)
    _next_states = states[:, 1:][actives]
    _next_obs = obs[:, 1:][actives]

    return _states, _obs, _actions, _rewards, _masks, _next_states, _next_obs, is_inits


def evaluate(agent, env, environment="mujoco", num_evaluation=10, max_steps=None):
    episode_rewards = []
    if max_steps is None and environment == "mujoco":
        max_steps = 1000
    assert max_steps != None

    for _ in range(num_evaluation):
        obs, _, _ = env.reset()
        episode_reward = 0
        for _ in range(max_steps):

            actions = agent.step((np.array(obs)).astype(np.float32))
            action = actions.numpy()
            
            next_obs, _, reward, done, _, _ = env.step(action)
            episode_reward += reward[0,0,0]

            if done[0,0]:
                break
            obs = next_obs
        episode_rewards.append(episode_reward)
        
    return np.mean(episode_rewards)