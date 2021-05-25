import numpy as np
import tensorflow as tf
from gym import spaces
from pommerman.agents import ActorCritic
from pommerman.agents import BaseAgent
from pommerman import constants
from pommerman.agents.action_prune import get_filtered_actions

class ActorCriticAgent(BaseAgent):
    def __init__(self):
        super(ActorCriticAgent, self).__init__()
        self.model = ActorCritic()
        self.current_state = None

        self.last_board = None
        self.prev_obs = [None,None]

    def __call__(self, state):
        # Check wether state is batched
        try:
            if isinstance(state[0], dict):
                state = self.obs_to_net_in(state[0])
                state = tf.expand_dims(state,0)
        except:
            if isinstance(state, dict):
                state = self.obs_to_net_in(state)
                state = tf.expand_dims(state,0)
        return self.model(state)

    def get_state(self):
        return self.current_state

    def get_input_shape(self, env):
        obs = env.reset()
        net_in = self.obs_to_net_in(obs)
        net_in = tf.expand_dims(net_in,0)
        return net_in.shape

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def set_model(self, model):
        self.model = model

    def obs_to_net_in(self, obs):
        """
            Takes the observation dictionary and turns value into feature planes of same shape
        """

        # Handle order of the keys
        key_list = [
            "bomb_life",
            "bomb_blast_strength",
            "bomb_moving_direction",
            "flame_life",
            "blast_strength",
            "can_kick",
            "ammo",
            #"game_type",
            #"game_env",
            #"step_count",
            #"alive",
            #"position",
            #"teammate",
            #"enemies",
            #"message",
            "board"
        ]

        type_dict = {
            # Full board to break down
            "board":"full_board",

            # Board type values:
            "bomb_blast_strength":"board",
            "bomb_life":"board",
            "bomb_moving_direction":"board",
            "flame_life":"board",

            # Single value type values:
            "blast_strength":"val",
            "can_kick":"val",
            "ammo":"val",
            "game_type":"val",
            "game_env":"val",
            "step_count":"val",

            # Special type values:
            "alive":None,
            "position":"pos",
            "message":"pos",
            "teammate":"mate",
            "enemies":"enemy"
        }

        # Rating of items (Prefere power-ups, avoid flames)
        desire_dict = {
            constants.Item.ExtraBomb.value:0,
            constants.Item.Kick.value:0,
            constants.Item.IncrRange.value:0,            
            constants.Item.Wood.value:1,
            constants.Item.Passage.value:2,
            constants.Item.Fog.value:3,
            constants.Item.Rigid.value:5,
            constants.Item.Agent0.value:6,
            constants.Item.Agent1.value:6,
            constants.Item.Agent2.value:6,
            constants.Item.Agent3.value:6,
            constants.Item.AgentDummy.value:6,
            constants.Item.Bomb.value:7,
            constants.Item.Flames.value:8
        }

        board_shape = obs['board'].shape
        agents = [constants.Item.Agent0,constants.Item.Agent1,constants.Item.Agent2,constants.Item.Agent3,constants.Item.AgentDummy]
        enemies = obs['enemies']
        mate = obs['teammate']
        alive = obs['alive']
        player = [constants.Item(e) for e in alive]
        for e in enemies:
            desire_dict[e.value] = 4
            try:
                player.remove(e)
            except:
                pass
        
        try:
            player.remove(mate)
        except:
            pass

        net_in = np.zeros((*board_shape,len(key_list)+len(constants.Item)-1))

        for idx, key in enumerate(key_list):
		    # Determine current value_type to handle value accordingly
            value = obs[key]
            value_type = type_dict[key]

            if value_type == 'full_board':
                # Break board down in its different features
                desire_plane = np.zeros(shape=board_shape, dtype=np.float32)
                enemy_plane = np.zeros(shape=board_shape, dtype=np.float32)
                mate_plane = np.zeros(shape=board_shape, dtype=np.float32)
                self_plane = np.zeros(shape=board_shape, dtype=np.float32)
                i=0
                # One-hot encode every item
                for e in constants.Item:
                    feature_plane = np.where(value==e.value, np.ones(shape=board_shape, dtype=np.float32), np.zeros(shape=board_shape, dtype=np.float32))
                    if e in agents:
                        if e in enemies:
                            enemy_plane += feature_plane
                        if e == mate:
                            mate_plane += feature_plane
                        if e in player:
                            self_plane = feature_plane
                    else:
                        net_in[:,:,idx+i] = feature_plane
                        i+=1
                        
                    # Encode desired positions
                    desire_plane += feature_plane*desire_dict[e.value]

                if self.last_board is None:
                    self.last_board = desire_plane.copy()
                net_in[:,:,idx+i] = enemy_plane
                i+=1
                net_in[:,:,idx+i] = mate_plane
                i+=1
                net_in[:,:,idx+i] = self_plane
                i+=1
                net_in[:,:,idx+i] = self.last_board
                net_in[:,:,-1] = desire_plane
                self.last_board = desire_plane.copy()

            else:
                if value_type == 'board':
                    # Value already board_shape
                    feature_plane = np.array(value, dtype=np.float32)

                elif value_type == 'val':
                    # Fill array of board_shape with value
                    feature_plane = np.zeros(shape=board_shape, dtype=np.float32)
                    feature_plane.fill(np.float32(value))
                
                elif value_type == 'pos':
                    # Mark position in zero plane
                    feature_plane = np.zeros(shape=board_shape, dtype=np.float32)
                    feature_plane[value[0],value[1]] = 1

                elif value_type == 'mate':
                    # Fill array of board_shape with mate value for alive mate
                    feature_plane = np.zeros(shape=board_shape, dtype=np.float32)
                    if value.value in obs['alive']:
                        feature_plane.fill(np.float32(value.value))

                elif value_type == 'enemy':
                    # Sum up enemy values for alive enemies
                    feature_plane = np.zeros(shape=board_shape, dtype=np.float32)
                    for val in value:
                        if val.value in obs['alive']:
                            feature_plane += val.value

                else:
                    # Create zero plane if value is not handled
                    feature_plane = np.zeros(shape=board_shape, dtype=np.float32)

                net_in[:,:,idx] = feature_plane
        
        return net_in
        
    def act(self, state, action_space = spaces.Discrete(6), training = False):
        # Check wether state is batched
        try:
            if isinstance(state[0], dict):
                state = state[0]
                net_in = self.obs_to_net_in(state)
        except:
            if isinstance(state, dict):
                net_in = self.obs_to_net_in(state)

        self.current_state = net_in

        network_out = self.model(tf.expand_dims(net_in,0))
        probs = network_out["policy"]
        # print(probs)
        # print(network_out["value_estimate"])
        allowed_actions = get_filtered_actions(state,(self.prev_obs[-2],self.prev_obs[-1]))
        allowed_probs = tf.gather(probs,allowed_actions)
        allowed_probs /= tf.reduce_sum(allowed_probs)

        self.prev_obs.append(state)
        
        if not training:
            action = allowed_actions[tf.argmax(allowed_probs)]
            return action
        else:
            action = np.random.choice(allowed_actions, p=allowed_probs.numpy())

            log_prob = tf.math.log(probs)
            v_estimate = network_out["value_estimate"]
            return action, log_prob[action], v_estimate

    def flowing_log_prob(self, feature_state_batch, action_batch):
        network_out = self.model(feature_state_batch)
        probs = network_out["policy"]
        action_batch = tf.cast(action_batch, dtype=tf.int32)
        log_prob = tf.math.log(probs)
        entropy = -tf.reduce_sum(probs * log_prob, axis=-1)        
        return tf.convert_to_tensor([log_prob[i][action] for i,action in enumerate(action_batch)]), entropy

    def save(self, path):
        self.model.save(path)

    def episode_end(self, reward):
        self.current_state = None
        self.prev_obs = [None,None]