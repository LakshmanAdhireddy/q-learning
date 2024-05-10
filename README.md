# Q Learning Algorithm


## AIM

To develop a Python program to find the optimal policy for the given RL environment using Q-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT

Develop a Python program to derive the optimal policy using Q-Learning and compare state values with Monte Carlo method.

## Q LEARNING ALGORITHM

Step 1: Initialize Q-table and hyperparameters.

Step 2: Choose an action using the epsilon-greedy policy and execute the action, observe the next state, reward, and update Q-values and repeat until episode ends.

Step 3: After training, derive the optimal policy from the Q-table.

Step 4: Implement the Monte Carlo method to estimate state values.

Step 5: Compare Q-Learning policy and state values with Monte Carlo results for the given RL environment.

~~~
Developed by:Lakshman
Reg no:212222240001
~~~

## Q LEARNING FUNCTION

~~~python

print('Name: Lakshman')
print('Reg.No: 212222240001')
def q_learning(env, 
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    select_action = lambda state,Q,epsilon: np.argmax(Q[state]) if np.random.random()>epsilon else np.random.randint(len(Q[state]))
    alphas=decay_schedule(init_alpha,min_alpha,alpha_decay_ratio,n_episodes)
    epsilons = decay_schedule(init_epsilon,min_epsilon,epsilon_decay_ratio,n_episodes)
    for e in tqdm(range(n_episodes),leave=False):
      state,done=env.reset(),False
      action=select_action(state,Q,epsilons[e])
      while not done:
        
        action=select_action(state,Q,epsilons[e])
        next_state,reward,done,_=env.step(action)
        td_target=reward+gamma*Q[next_state].max()*(not done)
        td_error=td_target-Q[state][action]
        Q[state][action]=Q[state][action]+alphas[e]*td_error
        state=next_state
      Q_track[e]=Q
      pi_track.append(np.argmax(Q,axis=1))
    V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return Q, V, pi, Q_track, pi_track
~~~

## OUTPUT:
![Screenshot 2024-05-10 083711](https://github.com/LakshmanAdhireddy/q-learning/assets/118707265/0c7abecd-39bf-46c1-aed4-8f995738b95d)

![Screenshot 2024-05-10 084125](https://github.com/LakshmanAdhireddy/q-learning/assets/118707265/618b071e-bf93-4845-ae6c-90be0d6737b4)

![Screenshot 2024-05-10 083847](https://github.com/LakshmanAdhireddy/q-learning/assets/118707265/ec28d6b1-e2d9-4247-9d48-791b1bf8a028)

![Screenshot 2024-05-10 083923](https://github.com/LakshmanAdhireddy/q-learning/assets/118707265/6520095b-cc38-411b-b630-df5c09e778b0)

![Screenshot 2024-05-10 083951](https://github.com/LakshmanAdhireddy/q-learning/assets/118707265/f68e0742-8a1e-4c12-88c3-347479ca12fe)

## RESULT:

Therefore a python program has been successfully developed to find the optimal policy for the given RL environment using Q-Learning and compared the state values with the Monte Carlo method.
