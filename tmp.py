import tensorflow as tf

# Create some variables.
v1 = tf.get_variable("v1", shape=[3], initializer=tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer=tf.zeros_initializer)
v3 = tf.get_variable("v3", shape=[4], initializer=tf.zeros_initializer)

v2_3 = v2[:3]
v_stack_0 = tf.stack([v1, v2_3], axis=0, name='111')
print(tf.get_default_graph().get_operations())
a = tf.get_default_graph().get_operation_by_name('111')
print(a.values())
v_stack_1 = tf.stack([v1, v2_3], axis=0, name='111')
tf.reduce_sum(v_stack_0)
tf.reduce_sum(v_stack_0)
'''tf.reduce_sum(v_stack_1)
v_stack_0 + v_stack_1
v_stack_0 + v_stack_1'''

for op in tf.get_default_graph().get_operations():
    print(op.name, op.values(), sep='\n\t')
    print(op.inputs)
