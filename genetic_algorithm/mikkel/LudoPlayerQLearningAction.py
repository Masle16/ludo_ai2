import random
import csv
import os
import numpy as np
import pandas as pd
import time
import cProfile
from pyludo.utils import *
from pyludo.LudoGame import LudoState


class LudoPlayerQLearningAction:
    ####### Class variables ########
    name = 'QLearning'
    # Token states
    homed = 0
    normal = 1
    stacked = 2 # safe if player is stacked
    globe = 3 # safe on globe except other players home globe
    vulnerable = 4 # other players token can hit your token
    globe_home = 5 # other players globe at home
    goal = 6 # when token is inside goal
    star = 7 # when on a star 
    end_game_area = 8 # Last 5 spaces into goal

    # Rewards
    r_normal = 0
    r_safe = 0
    r_got_vulnerable = -1
    r_knock_home = 1
    r_suicide = 0
    r_star_jump = 0
    r_move_onto_board = 1

    r_moved_end_game_token = 0
    r_end_game_safe = 1
    r_one_token_win = 1
    r_win = 1
    
    # Actions
    actions = np.array([0,1,2,3,4]) #np.array([0,1,2,3,4,5,6,7,8]) # 5 actions, [moved_out, into_goal, send_opp_home, send_self_home, move_token] 

    def __init__(self, Parameters, chosenPolicy="epsilon greedy", QtableName=None, RewardName=None):

        
        self.__chosenPolicy = chosenPolicy
        
        # Only used for GA
        self.chromosome = Parameters

        # Parameters
        self.__epsilon = Parameters[0]
        self.__discount_factor = Parameters[1]
        self.__alpha = Parameters[2]

        # Data logging rewards and QTable
        if RewardName is not None:
            self.Reward_save_name = RewardName + f'_e-{self.__epsilon}_d-{self.__discount_factor}_a-{self.__alpha}.csv'
        if QtableName is not None:
            self.Qtable_save_name = QtableName + f'_e-{self.__epsilon}_d-{self.__discount_factor}_a-{self.__alpha}.csv'

        if(QtableName is None):
            self.__QTable = dict()
        else:
            self.__QTable = self.readQTable()

        self.total_reward = 0.0
        self.rewards = []

    def __will_send_opponent_home(self, opponent_states, opponent_next_states):
        """
        Function for evaluating if possible
        to knock opponent home
        """
        for player_idx, opponent_next_state in enumerate(opponent_next_states):
            if opponent_next_state is False:
                continue
            opponent_state = opponent_states[player_idx]
            if np.sum( opponent_state[:] == -1 ) < np.sum( opponent_next_state[:] == -1 ):
                return True
        return False


    def __valid_token_moves(self, state, next_state, token_id):
        """
        Finds valid moves for a token
        """
        if next_state == False:
            return [False] * (len(LudoPlayerQLearningAction.actions) - 1) # State is not described with move_token

        current_pos_token = state.state[0][token_id]
        next_pos_token = next_state.state[0][token_id]

        current_opponent_states = state.state[1:]
        next_opponent_states = next_state.state[1:]

        moved_out = (current_pos_token == -1) and (next_pos_token != -1)
        into_goal = (current_pos_token != 99) and (next_pos_token == 99)

        send_opp_home = will_send_opponent_home(state, next_state) #self.__will_send_opponent_home(np.array(current_opponent_states), np.array(next_opponent_states))
        send_self_home = will_send_self_home(state, next_state) #(current_pos_token != -1) and (next_pos_token == -1)

       # enter_safe_zone = (current_pos_token <= 51) and (next_pos_token > 51)
       # on_star = (star_jump(current_pos_token) == 0) and (star_jump(next_pos_token) != 0)
       # on_glob = (is_globe_pos(current_pos_token) == False) and (is_globe_pos(next_pos_token) == True)
       # vulnerable = (token_vulnerability(state, token_id, 0) == False) and (token_vulnerability(next_state, token_id, 0) == True)
       

        reduced_state = [moved_out, into_goal, send_opp_home, send_self_home] #[moved_out, into_goal, send_opp_home, send_self_home, enter_safe_zone, on_star, on_glob, vulnerable] # True if action is valid

        return reduced_state

    def __get_actions(self, state, next_states):
        """
        Converts whole state representation of one player,
        to a simpler one. State is a representation of what
        the player can do. E.g. move_out 
        """
        num_of_actions = len(LudoPlayerQLearningAction.actions) - 1 # State is not described with move_token
        val_tok_mov = np.zeros((4, num_of_actions))
        for token_id in range(4):
            val_tok_mov[token_id] = self.__valid_token_moves(state, next_states[token_id], token_id)

        reduced_state = np.logical_or.reduce((val_tok_mov[0,:], val_tok_mov[1,:], val_tok_mov[2,:], val_tok_mov[3,:]))

        return reduced_state, val_tok_mov

    def __action_to_token(self, action, next_states, val_tok_mov):
        """
        Maps from combined action chosen based on
        the minimal state representation to which token
        to move. Returns empty if no token can do action
        If move than one token can do the same move, 
        return a random token.
        """
        array_length = len(LudoPlayerQLearningAction.actions)
        move_token = LudoPlayerQLearningAction.actions[array_length - 1]

        if (int)(action) == move_token: #  move_token action returns first valid token to move
            for token_id, next_state in enumerate(next_states):
                if next_state is not False:
                    return token_id

        valid_tokens = np.argwhere(val_tok_mov[:,int(action)] == True).squeeze()
        if valid_tokens.size > 1:
            return np.random.choice(valid_tokens)
        else:
            return valid_tokens



    # STATE REPRESENTATION #
    def __getTokenState(self, state, player_num):
        """
        Converts whole state representation of one player,
        to a simpler one
        """
        playerState = np.array(state.state[player_num])
        tokenState = np.copy(playerState)

        # Set all to normal if no criteria is true
        tokenState[:] = LudoPlayerQLearningAction.normal

        # Homed
        tokenState[playerState == -1] = LudoPlayerQLearningAction.homed
        
        # Vulnerable
        tmp = [token_vulnerability(state, token_id, player_num) > 0 for token_id in range(4)]
        tokenState[tmp] = LudoPlayerQLearningAction.vulnerable

        # Stacked
        tmp = is_stacked(np.array(state.state), player_num)
        tokenState[tmp] = LudoPlayerQLearningAction.stacked

        # On globe
        tmp = [is_globe_pos(token) for token in playerState]
        tokenState[tmp] = LudoPlayerQLearningAction.globe

        # On star
        tmp = [star_jump(token) > 0 for token in playerState]
        tokenState[tmp] = LudoPlayerQLearningAction.star

        # On other players home globe
        tmp = is_on_opponent_globe(np.array(state.state), player_num)
        tokenState[tmp] = LudoPlayerQLearningAction.globe_home

        # Token end game last 5 space into goal
        tokenState[playerState >= 52] = LudoPlayerQLearningAction.end_game_area

        # Token inside goal
        tokenState[playerState == 99] = LudoPlayerQLearningAction.goal

        return tokenState

    def __calc_cum_reward(self, token_to_move, next_states):
        min_val = -1*4
        max_val = 99*4

        next_state_sum = np.sum(next_states[token_to_move][0])
        
        # Get the opponent who is most ahead by finding sum of the state
        opponents_next_state_sum = np.sum(next_states[token_to_move][1:])
        oppenent_ahead = np.max(opponents_next_state_sum)

        diff_state_sum = next_state_sum - oppenent_ahead

        return (diff_state_sum - min_val)/(max_val - min_val)

    def __calc_cum_mean_reward(self, token_to_move, next_states):
        """
        Calculates normalized cumulative reward based on
        all the opponents tokens meaned
        """
        min_val = -1
        max_val = 99

        next_state_sum = np.mean(np.sum(next_states[token_to_move][0]))
        
        # Get the opponent who is most ahead by finding sum of the state
        opponents_next_state_sum = np.sum(next_states[token_to_move][1:])
        opponents_mean = np.mean(np.mean(opponents_next_state_sum))

        diff_state_sum = next_state_sum - opponents_mean

        return (diff_state_sum - min_val)/(max_val - min_val)

    
    def __win_only_reward(self, token_to_move, next_states):
        next_state = next_states[token_to_move].state[0]
        if np.all(np.array(next_state) == 99):
            return 1
        else:
            return 0

    def append_reward(self):
        self.rewards.append(self.total_reward)
        self.total_reward = 0

    # MAKE Q-TABLE #

    def __updateQTable(self, tokenState, qValue):
        # Update dictionary
        strTokenState = str(tokenState)

        if (strTokenState in self.__QTable):
            tmpQValue = self.__QTable[strTokenState]
            self.__QTable[strTokenState] = np.add(qValue, tmpQValue)  
        # Make new entry
        else:
            self.__QTable[strTokenState] = qValue

    def saveQTable(self):
        if self.__chosenPolicy is not 'greedy':
            csv_writer = csv.writer(open(self.Qtable_save_name, "w", newline=''))
            for key, val in self.__QTable.items():
                csv_writer.writerow((key, val))
            print("QTable saved succefully")

    def readQTable(self):
        tmpQTable = dict()
        if os.path.isfile(self.Qtable_save_name):
            read = csv.reader(open(self.Qtable_save_name))
            i = 0
            for row in read:
                i = i + 1
                state, QVal = row
                QVal = np.fromstring(QVal[1:-1], sep=' ')
                tmpQTable[state] = np.array(QVal)
            print("QTable read succefully. Found " + str(i) + " states")
        else:
            print ("QTable file not found, making a new")
        
        return tmpQTable

    def saveReward(self):
        if self.__chosenPolicy is not 'greedy':
            with open(self.Reward_save_name, 'a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                for reward in self.rewards:
                    csv_writer.writerow([reward])

    def printQTable(self):
        print(self.__QTable)
      #  print("argmax", np.argmax(self.__QTable[str(np.array([0, 0, 0, 0]))]))
        pass


    # MAKE POLICIES #

    def policies(self, QTable, epsilon, state, next_states, action_to_do): # Inspiration from https://www.geeksforgeeks.org/q-learning-in-python/?fbclid=IwAR1UXR88IuJBhhTakjxNq_gcf3nCmJB0puuoA46J8mZnEan_qx9hhoFzhK8
        """ 
        Creates an epsilon-greedy policy based 
        on a given Q-function and epsilon. 
        
        Returns a function that takes the state 
        as an input and returns the probabilities 
        for each action in the form of a numpy array  
        of length of the action space(set of possible actions). 
        """
        num_actions = len(LudoPlayerQLearningAction.actions) 
        def epsilonGreedyPolicy(tokenState): 
            tmpTokenState = str(tokenState)
            
            
            valid_actions = np.append(action_to_do, True) # the True appended is move_token
            valid_act_len = len(np.where(valid_actions==True)[0])

            Action_probabilities = np.ones(num_actions, dtype = float) * epsilon / valid_act_len  # divides probability based on number of valid actions and epsilon (each 0.025 if 4 actions)       
            Action_probabilities = np.multiply(Action_probabilities, valid_actions)

            # If same values in QTable choose random valid action 
            best_action = np.argmax(QTable[tmpTokenState]) # Find index of action which gives highest QValue

            # Check if valid action else find new best action
            if not valid_actions[best_action]:
                actions = np.argsort(-QTable[tmpTokenState]) # descending order of action values
                for i in range(len(valid_actions)):
                    if valid_actions[actions[i]]:
                        best_action = actions[i]
                        break

            Action_probabilities[best_action] += (1.0 - epsilon) # Assigns rest probability to best action so probability sums to 1

            return Action_probabilities 

        def greedyPolicy(tokenState):
            tmpTokenState = str(tokenState)

            valid_actions = np.append(action_to_do, True) # the True appended is move_token

            Action_probabilities = np.zeros(num_actions, dtype = float)

            best_action = np.argmax(QTable[tmpTokenState]) # Find index of action which gives highest QValue
            # Check if valid action else find new best action
            if not valid_actions[best_action]:
                actions = np.argsort(-QTable[tmpTokenState]) # descending order of action values
                for i in range(len(valid_actions)):
                    if valid_actions[actions[i]]:
                        best_action = actions[i]
                        break

            Action_probabilities[best_action] += 1.0
            return Action_probabilities


        if(self.__chosenPolicy == "epsilon greedy"):
            return epsilonGreedyPolicy 
        if(self.__chosenPolicy == "greedy"):
            return greedyPolicy


    # Q LEARNING #
    def QLearning(self, state, next_states):  # Inspiration from https://www.geeksforgeeks.org/q-learning-in-python/?fbclid=IwAR1UXR88IuJBhhTakjxNq_gcf3nCmJB0puuoA46J8mZnEan_qx9hhoFzhK8
        # Convert statespace representation for current states
        tokenState = self.__getTokenState(state, 0) 

        actions, val_token_moves = self.__get_actions(state, next_states)
        
        # Creates entry if current state does not exists
        num_actions = len(LudoPlayerQLearningAction.actions) 
        self.__updateQTable(tokenState, np.zeros(num_actions))
        # Get probabilites based on initialized policy (chosenPolicy)
        policy = self.policies(self.__QTable, self.__epsilon, state, next_states, actions) # returns a policy function
        actionProbability = policy(tokenState)

        # Choose action based on the probability distribution
        action = np.random.choice( LudoPlayerQLearningAction.actions, p=actionProbability )
        token_to_move = self.__action_to_token(action, next_states, val_token_moves)

        # Find next state based on action and updates Q-table. 
        next_states_based_action = next_states[token_to_move] 
        nextTokenState = self.__getTokenState(next_states_based_action, 0) 

       

        # Cummulative reward
       # reward = self.__calc_cum_reward(token_to_move, next_states)
        #reward = self.__calc_cum_mean_reward(token_to_move, next_states)
       # self.total_reward += reward

        # Creates entry if nextTokenState does not exists
        if self.__chosenPolicy is not 'greedy': 
             # Static reward
            reward = self.__win_only_reward(token_to_move, next_states)
            self.total_reward += reward

            self.__updateQTable(nextTokenState, np.zeros(num_actions))

            # Update based on TD Update
            td_target = reward + self.__discount_factor * np.max(self.__QTable[str(nextTokenState)])
            td_delta = td_target - self.__QTable[str(tokenState)][action] 
            update_val = self.__alpha * td_delta 

            self.__QTable[str(tokenState)][action] += update_val

        return token_to_move

    def play(self, state, dice_roll, next_states):

       # cProfile.runctx('self.QLearning(state, next_states)', globals(), locals()) # Time profile for functions

        token_to_move = self.QLearning(state, next_states)

        return token_to_move # return number token want to move
