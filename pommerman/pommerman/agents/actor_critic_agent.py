import numpy as np
import tensorflow as tf
from pommerman.agents import ActorCritic
from pommerman.agents import BaseAgent

class ActorCriticAgent(BaseAgent):
    def __init__(self):
        super(ActorCriticAgent, self).__init__()
        self.model = ActorCritic()
        self.current_state = None

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

    def obs_to_net_in(self, obs):
        """
            Takes the observation dictionary and turns value into feature planes of same shape
        """

        # Handle order of the keys
        key_list = [
            "board",
            "bomb_blast_strength",
            "bomb_life",
            "bomb_moving_direction",
            "flame_life",
            "blast_strength",
            "can_kick",
            "ammo",
            #"game_type",
            #"game_env",
            #"step_count",
            #"alive",
            "position",
            #"teammate",
            "enemies",
            #"message",
        ]

        type_dict = {
            # Board type values:
            "board":"board",
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

        board_shape = obs['board'].shape

        net_in = np.zeros((*board_shape,len(key_list)))

        for idx, key in enumerate(key_list):
		    # Determine current value_type to handle value accordingly
            value = obs[key]
            value_type = type_dict[key]

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
        
    def act(self, state, action_space = None, training = False):
        # Check wether state is batched
        try:
            if isinstance(state[0], dict):
                net_in = self.obs_to_net_in(state[0])
        except:
            if isinstance(state, dict):
                net_in = self.obs_to_net_in(state)

        self.current_state = net_in
        network_out = self.model(tf.expand_dims(net_in,0))
        prob = network_out["policy"]   
        print(prob)   

        if not training:
            action = tf.argmax(prob, axis=-1)
            return action
        else:
            action = tf.squeeze(tf.random.categorical(tf.math.log([prob]),1)).numpy()
            log_prob = tf.math.log(prob[action])
            v_estimate = network_out["value_estimate"]
            return action, log_prob, v_estimate

    def flowing_log_prob(self, feature_state_batch, action_batch):
        network_out = self.model(feature_state_batch)
        prob = network_out["policy"]
        action_batch = tf.cast(action_batch, dtype=tf.int32)
        log_prob = tf.math.log(
            [prob[i][a] for i, a in zip(range(prob.shape[0]), action_batch)]
            )
        entropy = -tf.reduce_sum(prob * tf.math.log(prob), axis=-1)
        return log_prob, entropy

    def save(self, path):
        self.model.save(path)