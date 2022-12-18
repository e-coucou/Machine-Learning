import tensorflow as tf

U = tf.Variable(tf.random.normal((tensor_R.shape[0], latent_features), dtype=tf.float32), name='U')
V = tf.Variable(tf.random.normal((latent_features, tensor_R.shape[1]), dtype=tf.float32), name='V')

# ! tf.Variable required, reshape transforms automatically in tensor (non mutable)...
#------------------------------------------------------------------------------------
b_emb = np.random.normal(size=embeddings).reshape([1, -1])
b_emb = tf.Variable(b_emb, dtype=tf.float32, name='bias_latent')
b_batch = np.random.normal(size=tensor_R.shape[0]).reshape([-1, 1])
b_batch = tf.Variable(b_batch, dtype=tf.float32, name='bias_batch')
b_total = tf.Variable(np.mean(values, dtype=np.float32), name='bias_total')

# custom regularization of weights (U, V matrices, biases)
# add to loss function
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
def l2_reg_custom(U, V, lamb):
    return lamb * (tf.reduce_sum(tf.math.square(U)) + tf.reduce_sum(tf.math.square(V)) + tf.reduce_sum(tf.math.square(b_emb)) + tf.reduce_sum(tf.math.square(b_batch)) + b_total)

def l1_reg_custom(U, V, lamb):
    return lamb * (tf.reduce_sum(tf.math.abs(U)) + tf.reduce_sum(tf.math.abs(V)) + tf.reduce_sum(tf.math.abs(b_emb)) + tf.reduce_sum(tf.math.abs(b_batch)) + b_total)
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------


#optimizer = tf.optimizers.Adam(learning_rate=0.5)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.0000025, momentum=0.9)


trainable_weights = [
    U, V,
    b_emb,     
    b_batch,     
    b_total
    ]

@tf.function
def nmf_tf():
    
    with tf.GradientTape() as tape:
        tape.watch(trainable_weights)
        R_prime = tf.math.add(tf.math.add(tf.math.add(tf.matmul(U, V), b_emb), b_batch), b_total)
        R_prime_sparse = tf.gather(

            tf.reshape(R_prime, [-1]),
            indices_2[:, 0] * tf.shape(R_prime)[1] + indices_2[:, 1]

            )
        # loss without regularization
        #----------------------------
        #loss = tf.math.square(tf.math.subtract(R_prime_sparse, tensor_R.values))
        loss =  tf.reduce_sum(tf.math.square(tf.math.subtract(R_prime_sparse, tensor_R.values)))
        loss += l2_reg_custom(U, V, 1e-6)

    grads = tape.gradient(loss, trainable_weights)
    optimizer.apply_gradients(zip(grads, trainable_weights))
    
    return loss




losses_reg = []

for step in range(5000):
    loss = nmf_tf().numpy()
    if step == 0:
        losses_reg.append(loss)
    else:
        if loss <= 0.6:
            print(f'no significant changes at iteration {step}')
            break
        else:
            losses_reg.append(loss)
            if step % 100 == 0:
                print(f'iteration: {step}, loss: {np.round(loss.numpy(), 4)}')
