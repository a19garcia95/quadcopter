from keras import layers, models, optimizers
from keras import backend as K

class Actor:
    """
    This class defines the actor (policy) model for DDPG.

    """

    def __init__(self, task, lr_actor):
        """
        Initialize parameters and build model.

        """
        
        self.task = task
        
        self.state_size = task.state_size 
        self.action_size = task.action_size 
        
        self.action_low = self.task.action_low 
        self.action_high = self.task.action_high 
        self.action_range = self.action_high - self.action_low 

        self.lr_actor = lr_actor 
      
        self.build_model()

    def build_model(self):
        """
        Define a neural network for an actor (policy) model,
        i.e. the input is states and actions are returned.

        """
        
        states = layers.Input(shape=(self.state_size,), name='states')

        
        net = layers.Dense(units=32, activation='relu')(states)
        net = layers.BatchNormalization()(net)
        net = layers.Dropout(0.5)(net)
        net = layers.Dense(units=64, activation='relu')(net)
        net = layers.BatchNormalization()(net)
        net = layers.Dropout(0.5)(net)
        net = layers.Dense(units=32, activation='relu')(net)

        
        raw_actions = layers.Dense(units=self.action_size,
                                        activation='sigmoid',name='raw_actions')(net)

        
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
                                        name='actions')(raw_actions)

      
        self.model = models.Model(inputs=states, outputs=actions)

        
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        
        optimizer = optimizers.Adam(lr=self.lr_actor)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
                inputs=[self.model.input, action_gradients, K.learning_phase()],
                outputs=[],
                updates=updates_op)

class Critic:
    """
    This class defines the critic (value) model for DDPG.

    """

    def __init__(self, task, lr_critic):
        """
        Initialize parameters and build model.

        """
        
        self.task = task
       
        self.state_size = task.state_size 
        self.action_size = task.action_size 

        self.lr_critic = lr_critic 

       
        self.build_model()

    def build_model(self):
        """
        Define a neural network for a critic (value) model,
        i.e. the input is states and actions, and Q-value is returned
        (its gradient is also computed).

        """
        
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

      
        net_states = layers.Dense(units=32, activation='relu')(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Dropout(0.5)(net_states)
        net_states = layers.Dense(units=64, activation='relu')(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Dropout(0.5)(net_states)
        net_states = layers.Dense(units=32, activation='relu')(net_states)

       
        net_actions = layers.Dense(units=32, activation='relu')(actions)
        net_actions= layers.BatchNormalization()(net_actions)
        net_actions = layers.Dropout(0.5)(net_actions)
        net_actions = layers.Dense(units=64, activation='relu')(net_actions)
        net_actions= layers.BatchNormalization()(net_actions)
        net_actions = layers.Dropout(0.5)(net_actions)
        net_actions = layers.Dense(units=32, activation='relu')(net_actions)

        
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

       
        net = layers.Dropout(0.5)(net)
        net = layers.Dense(units=32, activation='relu')(net)

        
        Q_values = layers.Dense(units=1, name='q_values')(net)

        
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        
        optimizer = optimizers.Adam(lr=self.lr_critic)
        self.model.compile(optimizer=optimizer, loss='mse')

       
        action_gradients = K.gradients(Q_values, actions)

   
        self.get_action_gradients = K.function(
                        inputs=[*self.model.input, K.learning_phase()],
                        outputs=action_gradients)
