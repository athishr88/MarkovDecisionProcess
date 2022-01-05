# CSCI 5302 HW2
version = "v2020.9.25.0000"

import gym
import gym_mountaincar
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import itertools
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import math
import random


student_name = "Athish" # Set to your name
GRAD = True  # Set to True if graduate student


class TabularPolicy(object):
    def __init__(self, n_bins_per_dim, num_dims, n_actions):
        self.num_states = n_bins_per_dim ** num_dims
        self.num_actions = n_actions

        self._transition_function = np.zeros(shape=(self.num_states, self.num_actions, self.num_states))
        self._reward_function = np.zeros(shape=(self.num_states, self.num_actions, self.num_states))

        # Create data structure to store mapping from state to value
        # self._value_function[state] = state value
        self._value_function = np.zeros(shape=(self.num_states,))

        # Create data structure to store array with probability of each action for each state
        # self._policy[state] = [array of action probabilities]
        self._policy = np.random.uniform(0, 1, size=(self.num_states, self.num_actions))

    def get_action(self, state):
        '''
        Returns an action drawn from the policy's action distribution at state
        '''
        prob_dist = np.array(self._policy[state])
        assert prob_dist.ndim == 1

        # Sample from policy distribution for state
        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.multinomial.html
        idx = np.random.multinomial(1, prob_dist / np.sum(prob_dist))
        return np.argmax(idx)

    def set_state_value(self, state, value):
        '''
        Sets the value of a given state
        '''
        self._value_function[state] = value

    def get_state_value(self, state):
        '''
        Returns the value of a given state
        '''
        return self._value_function[state]

    def get_value_function(self):
        '''
        Returns the table representing the value function itself.
        Useful to do for storing a value function at the end of an iteration of VI or PI...
        '''
        return copy.deepcopy(self._value_function)

    def set_value_function(self, v):
        '''
        Sets value function with new matrix v
        '''
        self._value_function = copy.copy(v)

    def set_policy(self, state, action_prob_array):
        '''
        Sets the action probabilities for a given state
        '''
        self._policy[state] = copy.copy(action_prob_array)

    def get_policy(self, state):
        '''
        Returns the probability distribution over actions for a given state
        '''
        return self._policy[state]

    def get_policy_function(self):
        '''
        Returns the table representing the policy function itself.
        '''
        return copy.deepcopy(self._policy)

    def set_entire_policy_function(self, p):
        '''
        Sets value function with new matrix v
        '''
        self._policy = copy.deepcopy(p)

    def set_entire_transition_function(self, t):
        '''
        Sets the entire transition function
        '''
        self._transition_function = copy.deepcopy(t)

    def set_transition_function(self, state_idx, action_idx, next_state_idx, val):
        '''
        Sets the transition function entry for a (s,a,s') tuple.
        '''
        self._transition_function[state_idx, action_idx, next_state_idx] = val

    def T(self, state_idx, action_idx, next_state_idx):
        '''
        Gets the transition function entry for a (s,a,s') tuple.
        '''
        return self._transition_function[state_idx, action_idx, next_state_idx]

    def set_reward_function(self, state_idx, action_idx, next_state_idx, val):
        '''
        Sets the reward function entry for a (s,a,s') tuple.
        '''
        self._reward_function[state_idx, action_idx, next_state_idx] = val

    def R(self, state_idx, action_idx, next_state_idx):
        '''
        Gets the reward function entry for a (s,a,s') tuple.
        '''
        return self._reward_function[state_idx, action_idx, next_state_idx]

class DiscretizedSolver(object):
    def __init__(self, mode, num_bins=21, lookahead=-1):
        self._mode = mode
        self._lookahead_steps = lookahead
        assert mode in ['nn', 'linear', 'lookahead']
        self._num_bins = num_bins

        self.env = gym.make("mountaincar5302-v0") # Problem Environment
        self.env_name = 'MountainCar'
        start_state = self.env.reset()

        self.state_lower_bound = self.env.observation_space.low
        self.state_upper_bound = self.env.observation_space.high
        self.bin_sizes = (self.state_upper_bound - self.state_lower_bound) / self._num_bins
        self.num_dims = self.state_lower_bound.shape[0]
        self.gamma = 0.99
        self.solver = TabularPolicy(self._num_bins, self.num_dims, self.env.action_space.n)
        self.performance_history = [] # List of cumulative rewards from successive runs of the policy

        self.populate_transition_and_reward_funcs() # Fill in our own T and R functions by using the simulator for the environment!



    def populate_transition_and_reward_funcs(self):
        '''
        Making use of self.add_transition, this function sets up the last of our discretized MDP variables
        T and R by iterating over all of our discrete states and filling in the appropriate values.
        '''

        # Iterate over all state/action combinations
        # Call self.add_transition for each
        for state_idx in range(self.solver.num_states):
            for action_idx in range(self.solver.num_actions):
                reward, next_states_and_probs = self.add_transition(state_idx, action_idx)
                for pair in next_states_and_probs:
                    self.solver.set_reward_function(state_idx, action_idx, pair[0], reward)
                    self.solver.set_transition_function(state_idx, action_idx,pair[0], pair[1])


        # raise NotImplementedError


    def add_transition(self, state_idx, action_idx):
        '''
        Sets the discretized MDP transition and reward values for a given state-action pair.
        state_idx: int corresponding to the 'from' state
        action_idx: int corresponding to the action taken
        '''

        # MountainCar is deterministic, so you do not need to sample multiple times for each state/action pair
        # in order to find out T and R.

        # HINT: You can set self.env.state to the state you wish to simulate (e.g., self.env.state = [0.001, 0.5])
        # HINT: You can use the self.env.step function to simulate an action. It returns 4 values: next_state, reward, environment_complete, info.
        # HINT: Keep in mind that we're setting T and R for our approximate, discretized MDP! You'll need to map the continuous state onto your discrete state space somehow.

        # Your code here. Remember to set_transition_function and set_reward_function on self.solver to store the values you compute!

        self.env.state = self.get_coordinates_from_state_index(state_idx)
        next_state, reward, done, info = self.env.step(action_idx)
        next_state_and_prob = self.get_discrete_state_probabilities(next_state)

        return reward, next_state_and_prob
        # raise NotImplementedError



    def get_discrete_state_probabilities(self, continuous_state_vector):
        '''
        Given a continuous state (e.g., [1.2,-0.3]), 
        return a list of (discrete state index, probability of being in this state) tuples

        For mode=='nn', this list will only have one element, consisting of the **state index**
        closest to continuous_state_vector e.g.,: [(6, 1.0)] to indicate state index 6 with 100% probability.

        For mode=='linear', this list will have multiple elements, indicating the nearest discrete states.
        Specifically, it should have 2^(state_dimension) elements, each containing (state index, probability). Mountain Car is a 2D problem.
        
        To build intuition, for a 1-dimensional problem this would indicate interpolation between two states (e.g., for a 1D continuous state space with 3 discrete states S1 S2 S3, given continuous state "Sx":  S1-----Sx---S2----------S3, it would return a list containing state indices 1 and 2 along with the probability of each, but not state 3.)
        '''
        # HINT: Computing distances to every state in your discrete state space is going to be slow! Try to come up with a faster solution.

        if self._mode == 'nn':
            # Find the closest discrete state to continuous_state_vector
            state_index = self.get_state_index_from_coordinates(continuous_state_vector)
            return [[state_index, 1]]
            # raise NotImplementedError
        elif self._mode == 'linear':
            # Find the 4 discrete states that surround continuous_state_vector, and assign
            # probability inversely proportionate to their distance from it.

            disc_state1_idx = self.get_state_index_from_coordinates(continuous_state_vector) # First state
            probable_states = [disc_state1_idx]
            if continuous_state_vector[1] + self.bin_sizes[1] < self.state_upper_bound[1]:
                probable_states.append(disc_state1_idx + 1)
            if continuous_state_vector[1] - self.bin_sizes[1] > self.state_lower_bound[1]:
                probable_states.append(disc_state1_idx - 1)
            if continuous_state_vector[0] + self.bin_sizes[0] < self.state_upper_bound[0]:
                probable_states.append(disc_state1_idx + self._num_bins)
            if continuous_state_vector[0] - self.bin_sizes[0] > self.state_lower_bound[0]:
                probable_states.append(disc_state1_idx - self._num_bins)

            distances = []
            for discrete_state_index in probable_states:
                disc_coordinates = self.get_coordinates_from_state_index(discrete_state_index)
                distance = math.sqrt(((disc_coordinates[0] - continuous_state_vector[0])/self.bin_sizes[0])**2 + ((disc_coordinates[1] - continuous_state_vector[1])//self.bin_sizes[1])**2)
                distances.append(distance)

            distribution = 1/np.array(distances)
            probabilities = distribution / np.sum(distribution)

            # return_matrix = np.zeros((len(probable_states), 2))
            return_matrix =[]

            # for x in range(len(probable_states)):
            #     return_matrix[x] = [probable_states[x], probabilities[x]]
            # print(return_matrix)

            for x in range(len(probable_states)):
                return_matrix.append([probable_states[x], probabilities[x]])
            # print(return_matrix)

            return return_matrix

            # raise NotImplementedError

    def compute_policy(self, max_iterations=150):
        '''
        Compute a policy and store it in self.solver (type TabularPolicy) using Value Iteration for max_iterations iterations.
        '''
        # HINT: Add the cumulative reward from self.solve() into self.performance_history each iteration
        #       Make sure to pick a reasonable value for the max_steps parameter (i.e., less than infinity)

        # Your Code Here -- You should be able to reuse most of your VI code from tabular_solution.py here
        # GRADs: Make sure to include code supporting the policy evaluation for n-step lookahead at the end of each iteration.
        #        You can use self.solver.set_policy_function and self.solve to get reward values to append to self.performance_history

        # HINT: Because this one can take a while to converge, set up a stopping criteria.
        #       If ||(v_i+1 - v_i)||_2 is small (i.e., distance between v_i+1 and v_i is small), the value function isn't updating much and you can safely stop iterating.

        # Remember to call self.solver.set_policy/value_function at the end!

        if self._mode == 'nn':
            v_i = self.solver.get_value_function()
            p_i = self.solver.get_policy_function()
            for iteration in range(max_iterations):
                print(iteration)
                v_iplus1 = np.zeros(v_i.shape)
                p_iplus1 = np.zeros(p_i.shape)
                for state_idx in range(self.solver.num_states):
                    action = self.solver.get_action(state_idx)
                    next_state_idx = self.add_transition(state_idx, action)[1][0][0]
                    reward = self.solver.R(state_idx, action, next_state_idx)
                    v_iplus1[state_idx] = reward + self.gamma * v_i[next_state_idx]

                    value_from_all_movements = []
                    for action_idx in range(self.solver.num_actions):
                        next_state_idx = self.add_transition(state_idx, action_idx)[1][0][0]
                        imm_reward = self.solver.R(state_idx, action_idx, next_state_idx)
                        value = imm_reward + self.gamma * v_i[next_state_idx]
                        value_from_all_movements.append(value)

                    best_action = np.argmax(value_from_all_movements)
                    p_iplus1[state_idx][best_action] = 1

                v_i = copy.deepcopy(v_iplus1)
                p_i = copy.deepcopy(p_iplus1)
                self.solver.set_value_function(v_i)
                self.solver.set_entire_policy_function(p_i)

        if self._mode == 'linear' and self._lookahead_steps < 0:
            v_i = self.solver.get_value_function()
            p_i = self.solver.get_policy_function()
            for iteration in range(max_iterations):
                print(iteration)
                v_iplus1 = np.zeros(v_i.shape)
                p_iplus1 = np.zeros(p_i.shape)
                for state_idx in range(self.solver.num_states):
                    value_from_all_movements = []
                    for action_idx in range(self.solver.num_actions):
                        reward, next_states_and_probs = self.add_transition(state_idx, action_idx)
                        next_states = []
                        probabilities = []
                        for pair in next_states_and_probs:
                            next_states.append(pair[0])
                            probabilities.append(pair[1])
                        value_of_nearest_states = []
                        for s_dash in next_states:
                            value_of_nearest_states.append(v_i[s_dash])

                        summed_v_s_dash = np.dot(value_of_nearest_states, probabilities)
                        value = reward + self.gamma * summed_v_s_dash
                        value_from_all_movements.append(value)
                    v_iplus1[state_idx] = np.max(value_from_all_movements)
                    best_action = np.argmax(value_from_all_movements)
                    p_iplus1[state_idx][best_action] = 1

                v_i = copy.deepcopy(v_iplus1)
                p_i = copy.deepcopy(p_iplus1)
                self.solver.set_value_function(v_i)
                self.solver.set_entire_policy_function(p_i)
        if self._lookahead_steps > 0:
            v_i = self.solver.get_value_function()
            p_i = self.solver.get_policy_function()
            for iteration in range(max_iterations):
                print(self.solve(max_steps=1000)[0])
                self.performance_history.append((self.solve(max_steps=1000)[0]))
                print(iteration)
                v_iplus1 = np.zeros(v_i.shape)
                p_iplus1 = np.zeros(p_i.shape)
                for state_idx in range(self.solver.num_states):
                    state_cord = self.get_coordinates_from_state_index(state_idx)
                    all_possible_sequences = list(itertools.product([0,1,2], repeat=self._lookahead_steps))
                    value_from_all_sequences = []
                    for sequence in all_possible_sequences:
                        cum_reward, next_state_cord = self.evaluate_action_sequence(state_cord, sequence)
                        next_states_and_probs = self.get_discrete_state_probabilities(next_state_cord)
                        next_states = []
                        probabilities = []
                        for pair in next_states_and_probs:
                            next_states.append(pair[0])
                            probabilities.append(pair[1])
                        value_of_nearest_states = []
                        for s_dash in next_states:
                            value_of_nearest_states.append(v_i[s_dash])

                        summed_v_s_dash = np.dot(value_of_nearest_states, probabilities)
                        value_of_sequence = cum_reward + (self.gamma**self._lookahead_steps)*summed_v_s_dash
                        value_from_all_sequences.append(value_of_sequence)
                    best_sequence_idx = np.argmax(value_from_all_sequences)
                    best_sequence = all_possible_sequences[best_sequence_idx]
                    best_action_for_state = best_sequence[0]
                    p_iplus1[state_idx][best_action_for_state] = 1
                    v_iplus1[state_idx] = np.max(value_from_all_sequences)
                v_i = copy.deepcopy(v_iplus1)
                p_i = copy.deepcopy(p_iplus1)
                self.solver.set_value_function(v_i)
                self.solver.set_entire_policy_function(p_i)
                # raise NotImplementedError



    def get_state_index_from_coordinates(self, continuous_state_vector):
        '''
        Returns the discrete state index of a given continuous state vector, 
        using the number of bins provided at instantiation
        '''
        bin_sizes = (self.state_upper_bound - self.state_lower_bound) / self._num_bins
        bin_location = ((continuous_state_vector - self.state_lower_bound) / bin_sizes).astype(int)

        return bin_location[0] * self._num_bins + bin_location[1]

    def get_coordinates_from_state_index(self, state_idx):
        '''
        Returns the continuous state vector for a given discrete state index, returning
        the coordinates from the "middle" of discrete state cell
        '''
        bin_sizes = (self.state_upper_bound - self.state_lower_bound) / self._num_bins
        coordinates = np.array([(state_idx//self._num_bins + 0.5) * bin_sizes[0], (state_idx % self._num_bins + 0.5) * bin_sizes[1]]) + self.state_lower_bound
        return coordinates

    def evaluate_action_sequence(self, start_state, action_sequence):
        '''
        Simulate forward N steps from start_state using action_sequence, then return the accumulated reward
        ex: evaluate_action_sequence([0.4,0.2], [1,1,1])
        '''
        cumulative_reward = 0

        self.env.state = start_state
        for action in action_sequence:
            next_state, reward, done, info = self.env.step(action)
            self.env.state = next_state
            cumulative_reward += reward

        return cumulative_reward, next_state


        # raise NotImplementedError

    def solve(self, visualize=False, max_steps=float('inf')):
        '''
        Reset the environment, then applies the solver's policy. 
        Returns cumulative reward and number of actions taken.
        '''

        finished = False
        cur_state = self.env.reset()

        cumulative_reward = 0
        num_steps = 0

        while finished is False and num_steps < max_steps:
            # Take an action in the environment
            action = None

            if self._lookahead_steps > 0:
                # Your Code Here -- Implement n-step lookahead here for n = self._lookahead_steps
                # Generate all n-length action sequences and evaluate them using the evaluate_action_sequence function above
                # Remember to record the start state before simulating so you can reset it after you've found an action sequence to execute the first action of.
                # HINT: itertools.product can help you find all the action sequences.


                action = self.solver.get_action(self.get_state_index_from_coordinates(cur_state))
                
                # raise NotImplementedError
            elif self._mode == 'nn':
                action = self.solver.get_action(self.get_state_index_from_coordinates(cur_state))
                # raise NotImplementedError
            elif self._mode == 'linear' and self._lookahead_steps < 0:
                action = self.solver.get_action(self.get_state_index_from_coordinates(cur_state))
                # raise NotImplementedError
            else:
                action = self.solver.get_action(self.get_state_index_from_coordinates(cur_state))

            # Execute Action            
            next_state, reward, finished, info = self.env.step(action)

            # Update state
            cur_state = next_state

            # Update cumulative reward
            cumulative_reward += reward

            # Update action counter
            num_steps += 1

            # Display new state of the world
            if visualize is True: self.env.render()

        return cumulative_reward, num_steps

    def plot_value_function(self, value_function, filename=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)
        ax = fig.axes[0]

        V = ((value_function - value_function.min()) / (value_function.max() - value_function.min() + 1e-6))
        V = V.reshape(self._num_bins, self._num_bins).T
        image = (plt.cm.coolwarm(V)[::-1,:,:-1] * 255.).astype(np.uint8)
        ax.set_title("Env: %s" % self.env_name )
        ax.imshow(image)
        width, height = fig.get_size_inches() * fig.get_dpi()
        canvas.draw()
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height),int(width),3)

        if filename is None:
            filename = "figures/%s_%s_%d.png" % (self.env_name, self._mode, self._num_bins)

        fig.savefig(filename)

        return image, fig


def plot_policy_curves(reward_histories, filename=None):
    plt.close()
    plt.clf()
    symbols = ['bs', 'r--', 'g^']
    for idx, reward_history in enumerate(reward_histories):
        plt.plot(range(len(reward_history)), reward_history, symbols[idx%len(symbols)])

    plt.xlabel("Iteration")
    plt.ylabel("Return")
    plt.title("Policy Iteration Performance" )

    plt.savefig(filename)
    return





if __name__ == '__main__':
    ########## Q3.a ############
    bin_counts = [21, 51, 151]
    for bin_count in bin_counts:
        mc_solver = DiscretizedSolver(mode='nn', num_bins=bin_count, lookahead=-1)
        start_time = time.time()
        mc_solver.compute_policy()
        elapsed_time = time.time() - start_time
        print("Computed Q3.a VI Policy with bin size %d in %g seconds" % (bin_count, elapsed_time))
        mc_solver.plot_value_function(mc_solver.solver.get_value_function())

    ########### Q3.b ############

    bin_counts = [21, 51, 151]
    for bin_count in bin_counts:
        mc_solver = DiscretizedSolver(mode='linear', num_bins=bin_count, lookahead=-1)
        start_time = time.time()
        mc_solver.compute_policy()
        elapsed_time = time.time() - start_time
        print("Computed Q3.b VI Policy with bin size %d in %g seconds" % (bin_count, elapsed_time))
        mc_solver.plot_value_function(mc_solver.solver.get_value_function())

    ########### Q4 ############
    if GRAD is True:
        n_step = [1,2,3]
        performance = []
        for lookahead in n_step:
            mc_solver = DiscretizedSolver(mode='linear', num_bins=51, lookahead=lookahead)
            start_time = time.time()
            mc_solver.compute_policy()
            elapsed_time = time.time() - start_time
            print("Computed Q4 Lookahead Policy in %g seconds" % elapsed_time)
            performance.append(mc_solver.performance_history)
        plot_policy_curves(performance, "figures/lookahead_mc.png")
