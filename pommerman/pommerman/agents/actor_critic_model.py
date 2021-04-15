import tensorflow as tf
from tensorflow.keras import Model

class ActorCritic(Model):
    def __init__(self, layers=[64,64,128,128,64,64,32,32], num_actions=6):
        super(ActorCritic, self).__init__()
        self.readin_layers = [
            tf.keras.layers.Conv2D(
                filters=num_filter,
                kernel_size=3,
                padding='same',
                activation='tanh'
            ) 
            for num_filter in layers]

        self.readin_layers.append(tf.keras.layers.GlobalAveragePooling2D())
        
        self.readout_policy = tf.keras.layers.Dense(units=num_actions,
                                                    activation='softmax',
                                                    name='Policy_readout'
                                                    )
                
        self.readout_value = tf.keras.layers.Dense(units=1,
                                                   activation='tanh',
                                                   name='Value_readout'
                                                   )

    @tf.function
    def call(self, observation):
        output = {}
        # Read in
        for layer in self.readin_layers:
            observation = layer(observation)

        # Actor
        output["policy"] = tf.squeeze(self.readout_policy(observation))

        # Critic
        output["value_estimate"] = tf.squeeze(self.readout_value(observation))

        return output

