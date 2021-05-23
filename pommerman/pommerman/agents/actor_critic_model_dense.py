import tensorflow as tf
from tensorflow.keras import Model

class ActorCriticDense(Model):
    def __init__(self, layers=[64,64,32], num_actions=6):
        super(ActorCriticDense, self).__init__()
        self.readin_layers = [
            tf.keras.layers.Dense(
                units=num_units,
                activation='relu'
            )
            for num_units in layers
        ]

        self.policy_layer = tf.keras.layers.Dense(units=64,
                                                  activation='relu')
        
        self.readout_policy = tf.keras.layers.Dense(units=num_actions,
                                                    activation=None,
                                                    name='Policy_readout'
                                                    )

        self.value_layer = tf.keras.layers.Dense(units=64,
                                                 activation='relu')

        self.readout_value = tf.keras.layers.Dense(units=1,
                                                   activation=None,
                                                   name='Value_readout'
                                                   )

    @tf.function
    def call(self, observation):
        output = {}
        
        # Read in
        for layer in self.readin_layers:
            observation = layer(observation)

        # Actor
        policy_pred = self.policy_layer(observation)
        output["policy"] = tf.squeeze(self.readout_policy(policy_pred))

        # Critic
        value_pred = self.value_layer(observation)
        output["value_estimate"] = tf.squeeze(self.readout_value(value_pred))

        return output

if __name__ == "__main__":
    model = ActorCriticDense()
    out=model(tf.random.uniform((1,262),0,15))
    model.summary()
    print(out)
    model = ActorCriticDense()
    out=model(tf.random.uniform((1,490),0,15))
    model.summary()
    print(out)