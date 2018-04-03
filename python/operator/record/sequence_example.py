
import tensorflow as tf


def make_sequence_example(length, source, target):
    return tf.train.SequenceExample(
        feature_lists=tf.train.FeatureLists(feature_list={
            'source': tf.train.FeatureList(feature=[
                tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))
                for v in source
            ]),
            'target': tf.train.FeatureList(feature=[
                tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))
                for v in target
            ])
        }),
        context=tf.train.Features(feature={
            'length': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[length])
            )
        })
    )


def parse_sequence_example(serialized):
    context, sequence = tf.parse_single_sequence_example(
        serialized=serialized,
        context_features={
             "length": tf.FixedLenFeature([], dtype=tf.int64)
        },
        sequence_features={
             "source": tf.FixedLenSequenceFeature([], dtype=tf.int64),
             "target": tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }
    )

    return (context['length'], sequence['source'], sequence['target'])
