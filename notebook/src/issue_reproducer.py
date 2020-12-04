import tensorflow as tf

class issueReproducer(tf.Module):

    def __init__(self, n_unit):
        """
        n_unit : int
        """
        self.variable = tf.Variable(tf.zeros((1, n_unit), dtype=tf.float32))
        self.l1 = tf.keras.layers.Dense(n_unit)
        self.optimizer = tf.optimizers.Adam()

    @tf.function
    def fit(self, tensor):
        """
        tensor : some tensor of shape : (1, n_unit)
        """
        with tf.GradientTape() as tape:
            output = self.l1(self.variable)
            loss = tf.reduce_sum(output - self.variable)
        grad = tape.gradient(loss, self.variable)
        self.optimizer.apply_gradients([(grad, self.variable)])

        return self.variable


if __name__ == "__main__":

    model = issueReproducer(5)
    tensor = tf.constant([[1,2,3,4,5]], dtype=tf.float32)

    variable = model.fit(tensor)

    print("Returned variable : {}".format(variable))