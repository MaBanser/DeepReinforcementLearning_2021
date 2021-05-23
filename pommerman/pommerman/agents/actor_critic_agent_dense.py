import numpy as np
import tensorflow as tf
from pommerman.agents import ActorCriticDense
from pommerman.agents import BaseAgent

class ActorCriticAgentDense(BaseAgent):
    def __init__(self):
        super(ActorCriticAgentDense, self).__init__()
        self.model = ActorCriticDense()
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

        net_in = []

        for key in key_list:
		    # Determine current value_type to handle value accordingly
            value = obs[key]
            value_type = type_dict[key]

            if value_type == 'board':
		        # Extend by flattened value
                net_in.extend(value.flatten())
            
            elif value_type == 'val':
                # Append single value
                net_in.append(np.float32(value))
            
            elif value_type == 'pos':
                # Extend by tuple value
                net_in.extend(value)

            elif value_type == 'mate':
                # Append mate value if alive
                if value.value in obs['alive']:
                    net_in.append(value.value)
                else:
                    net_in.append(0)


            elif value_type == 'enemy':
                # Sum up enemy values for alive enemies
                enem = 0
                for val in value:
                    if val.value in obs['alive']:
                        enem += val.value
                
                net_in.append(enem)

            else:
                # Append zero if value is not handled
                net_in.append(0)
        
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
        logits = network_out["policy"]
        print(logits)
        print(network_out["value_estimate"])

        if not training:
            action = tf.argmax(logits, axis=-1)
            return action
        else:
            action = tf.squeeze(tf.random.categorical([logits],1)).numpy()
            log_prob = tf.nn.log_softmax(logits)
            v_estimate = network_out["value_estimate"]
            return action, log_prob[action], v_estimate

    def flowing_log_prob(self, feature_state_batch, action_batch):
        network_out = self.model(feature_state_batch)
        logits = network_out["policy"]
        action_batch = tf.cast(action_batch, dtype=tf.int32)
        log_prob = tf.nn.log_softmax(logits)        
        prob = tf.nn.softmax(logits)
        entropy = -tf.reduce_sum(prob * log_prob, axis=-1)        
        return tf.convert_to_tensor([log_prob[i][action] for i,action in enumerate(action_batch)]), entropy

    def save(self, path):
        self.model.save(path)