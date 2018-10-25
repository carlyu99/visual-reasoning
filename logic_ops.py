import tensorflow as tf

trs = [0, 3, 6, 1, 4, 7, 2, 5, 8]
# swap = [2, 1, 0, 5, 4, 3, 8, 7, 6]
# trs_swap = [6, 3, 0, 7, 4, 1, 8, 5, 2]


def _w(x, axis=-1):
    _ = tf.maximum(-tf.log(tf.clip_by_value(1 - x, 1e-10, 1)), 1e-10)
    # _ = tf.maximum(1 / tf.clip_by_value(1 - x, 1e-10, 1) - 1, 1e-10)
    # x = tf.clip_by_value(x, 1e-10, 1 - 1e-10)
    # _ = tf.maximum(tf.log(1 - x) / tf.log(x), 1e-10)
    return _ / tf.reduce_sum(_, axis=axis, keepdims=True)


def _and(x, axis=-1):
    return tf.reduce_sum(x * _w(1 - x, axis=axis), axis=axis)


def _and2(xl):
    # print([op.name for op in xl], '_and2')
    return _and(tf.stack(xl, axis=-1))


def _or(x, axis=-1):
    return tf.reduce_sum(x * _w(x, axis=axis), axis=axis)


def _or2(xl):
    return _or(tf.stack(xl, axis=-1))


def _xor(x, axis=-1):
    return _or(x, axis=axis) - _and(x, axis=axis)


def _xor2(xl):
    return _xor(tf.stack(xl, axis=-1))


def _eq2(x, y):
    return 1 - _xor2([x, y])


def _rel(feats, func):
    return _and(tf.concat([
        _eq2(func([feats[0], feats[1]]), feats[2]),
        _eq2(func([feats[3], feats[4]]), feats[5]),
        _eq2(func([feats[6], feats[7]]), feats[8])], axis=1))


def rel_and(feats):
    return _or2([_rel(feats, _and2), _rel([feats[i] for i in trs], _and2)])


def rel_or(feats):
    return _or2([_rel(feats, _or2), _rel([feats[i] for i in trs], _or2)])


def rel_xor(feats):
    return _or2([_rel(feats, _xor2), _rel([feats[i] for i in trs], _xor2)])


def _rel_con_union(feats):
    union0 = _or2([feats[0], feats[1], feats[2]])
    union1 = _or2([feats[3], feats[4], feats[5]])
    union2 = _or2([feats[6], feats[7], feats[8]])
    return _and(tf.concat([_eq2(union0, union1), _eq2(union1, union2), _eq2(union2, union0)], axis=1))


def rel_con_union(feats):
    return _or2([_rel_con_union(feats), _rel_con_union([feats[i] for i in trs])])


def _moment(x, axis=-1):
    coef = tf.to_float(list(range(x.shape[axis])))
    for i in range(len(x.shape) - 1):
        coef = tf.expand_dims(coef, axis=0)
    return tf.reduce_sum(x * coef, axis=axis)


def _less(x, y, axis=-1):
    diff = _w(y, axis=axis) - _w(x, axis=axis)
    return tf.sigmoid(_moment(diff))


def _rel_progression(feats):
    return _and2([
        _less(feats[0], feats[1]),
        _less(feats[1], feats[2]),
        _less(feats[3], feats[4]),
        _less(feats[4], feats[5]),
        _less(feats[6], feats[7]),
        _less(feats[7], feats[8])
    ])


def rel_progression(feats):
    return _or2([_rel_progression(feats), _rel_progression([feats[i] for i in trs])])


def _one(x, axis=-1):
    # TODO: Is it necessary to set an extra constraint on one-hot-ness (for progression and consistent union)?
    pass


if __name__ == '__main__':
    data = [[1, 1, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0], [.3, .5, .7]]
    vec0 = [[1, 1, 0], [0, 0, 1]]
    vec1 = [[0, 0, .5], [0, .5, 0]]
    vec2 = [[.9, 1, .5], [0, .1, .5]]
    v0 = [[1, 0, 0]]
    v1 = [[0, 1, 0]]
    v2 = [[0, 0, .9]]
    inp0 = tf.placeholder(dtype=tf.float32, shape=(None, 3))
    inp1 = tf.placeholder(dtype=tf.float32, shape=(None, 3))
    inp2 = tf.placeholder(dtype=tf.float32, shape=(None, 3))
    # w1, w2 = _w(a)
    b = _and(inp0)
    c = _or(inp0)
    d = _xor(inp0)
    e = _less(inp0, inp1)
    f = rel_and([inp0, inp1, inp2, inp2, inp1, inp0, inp2, inp0, inp1])
    f1 = _rel([inp0, inp1, inp2, inp2, inp1, inp0, inp2, inp0, inp1], _and2)
    f2 = _rel([[inp0, inp1, inp2, inp2, inp1, inp0, inp2, inp0, inp1][i] for i in trs], _and2)
    g1 = _rel_progression([inp0, inp1, inp2, inp0, inp1, inp2, inp0, inp1, inp2])
    g2 = _rel_progression([[inp0, inp1, inp2, inp0, inp1, inp2, inp0, inp1, inp2][i] for i in trs])
    h1 = _rel_con_union([[inp0, inp1, inp2, inp0, inp1, inp2, inp0, inp1, inp2][i] for i in trs])
    # print(b.shape, c.shape)
    with tf.Session() as sess:
        fetch = sess.run([b, c, d], feed_dict={inp0: data})
        for _ in fetch:
            print(_)
        fetch = sess.run([e, f1, f2, f], feed_dict={inp0: vec0, inp1: vec1, inp2: vec2})
        for _ in fetch:
            print(_)
        fetch = sess.run([g1, g2, h1], feed_dict={inp0: v0, inp1: v1, inp2: v2})
        for _ in fetch:
            print(_)
