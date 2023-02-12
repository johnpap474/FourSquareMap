from AWAC import core

def Parameters():
    actor_critic = core.MLPActorCritic
    alpha = 0
    num_train_episodes = None
    done = None
    ac_kwargs = dict()
    seed = 0
    steps_per_epoch = 100
    epochs = 10000
    replay_size = 20000 #int(2000000)
    gamma = 0.99
    polyak = 0.995
    lr = 3e-4
    p_lr = 3e-4
    alpha = 0.0
    batch_size = 1024
    start_steps = 10000
    update_after = 0
    update_every = 50
    num_test_episodes = 10
    max_ep_len = 1000
    logger_kwargs = dict()
    save_freq = 1
    algo = 'SAC'

    return actor_critic, alpha, num_train_episodes,done, ac_kwargs, seed, steps_per_epoch, epochs, replay_size, gamma, polyak, lr, p_lr, alpha, batch_size, start_steps, update_every, update_after, num_test_episodes, max_ep_len, logger_kwargs, save_freq, algo
    """
    Soft Actor-Critic (SAC)
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.
        actor_critic: The constructor method for a PyTorch Module with an ``act``
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of
            observations as inputs, and ``q1`` and ``q2`` should accept a batch
            of observations and a batch of actions as inputs. When called,
            ``act``, ``q1``, and ``q2`` should return:
            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each
                                        | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                        | of Q* for the provided observations
                                        | and actions. (Critical: make sure to
                                        | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current
                                        | estimate of Q* for the provided observations
                                        | and actions. (Critical: make sure to
                                        | flatten this!)
            ===========  ================  ======================================
            Calling ``pi`` should return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                        | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                        | actions in ``a``. Importantly: gradients
                                        | should be able to flow back into ``a``.
            ===========  ================  ======================================
        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to SAC.
        seed (int): Seed for random number generators.
        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.
        epochs (int): Number of epochs to run and train agent.
        replay_size (int): Maximum length of replay buffer.
        gamma (float): Discount factor. (Always between 0 and 1.)
        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:
            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta
            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)
        lr (float): Learning rate (used for both policy and value learning).
        alpha (float): Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)
        batch_size (int): Minibatch size for SGD.
        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.
        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.
        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.
        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        logger_kwargs (dict): Keyword args for EpochLogger.
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
        """