import tensorflow as tf
from tensorflow.keras import Model

class ActorCritic(Model):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.readin_layers = [
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=3,
                    padding='same',
                    activation='tanh',
                    name='readin_1'
                ),
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=3,
                    padding='same',
                    activation='tanh',
                    name='readin_2'
                ),
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=3,
                    padding='same',
                    activation='tanh',
                    name='readin_3'
                ),
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=3,
                    padding='same',
                    activation='tanh',
                    name='readin_4'
                )
            ]


        self.policy_layers = [
            tf.keras.layers.Conv2D(
                filters=6,
                kernel_size=1,
                padding='valid',
                activation='tanh',
                name='Policy_layer_1'
            ),
            tf.keras.layers.Flatten()
        ]
        
        self.readout_policy = tf.keras.layers.Dense(units=6,
                                                    activation='softmax',
                                                    name='Policy_readout'
                                                    )

        self.value_layers = [
            tf.keras.layers.Conv2D(
                filters=6,
                kernel_size=1,
                padding='valid',
                activation='tanh',
                name='Critic_layer_1'
            ),
            tf.keras.layers.Flatten()
        ]

        self.readout_value = tf.keras.layers.Dense(units=1,
                                                   activation='tanh',
                                                   name='Critic_readout'
                                                   )

    @tf.function
    def call(self, observation):
        output = {}
        # print(observation.shape)

        for layer in self.readin_layers:
            observation = layer(observation)
            # print(layer)
            # print(observation.shape)

        policy_pred = observation
        value_pred = observation

        # Actor
        for layer in self.policy_layers:
            policy_pred = layer(policy_pred)
            # print(policy_pred.shape)

        output["policy"] = tf.squeeze(self.readout_policy(policy_pred))

        # Critic
        for layer in self.value_layers:
            value_pred = layer(value_pred)
            # print(value_pred.shape)

        output["value_estimate"] = tf.squeeze(self.readout_value(value_pred))

        return output

if __name__ == "__main__":
    model = ActorCritic()
    out=model(tf.random.uniform((1,11,11,21),0,15))
    model.summary()
    print(out)