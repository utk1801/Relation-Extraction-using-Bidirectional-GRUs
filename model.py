import tensorflow as tf
from tensorflow.keras import layers, models

from util import ID_TO_CLASS


class MyBasicAttentiveBiGRU(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyBasicAttentiveBiGRU, self).__init__()

        self.num_classes = len(ID_TO_CLASS)

        self.decoder = layers.Dense(units=self.num_classes)
        self.omegas = tf.Variable(tf.random.normal((hidden_size*2, 1)))
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        ### TODO(Students) START
        # ...
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.training = training
        self.gru_layer = layers.GRU(self.hidden_size,return_sequences=True)
        self.bidir = layers.Bidirectional(self.gru_layer, merge_mode='concat')

        ### TODO(Students) END

    def attn(self, rnn_outputs):
        ### TODO(Students) START

        M = tf.nn.tanh(rnn_outputs)
        dot_M = tf.tensordot(M,self.omegas,axes=1)
        alpha = tf.nn.softmax(dot_M,axis=1)
        reshaped_alpha = alpha * tf.ones((1, 2*self.hidden_size))
        r = tf.reduce_sum(rnn_outputs * reshaped_alpha,axis=1)

        output=tf.nn.tanh(r)
        ### TODO(Students) END

        return output

    def call(self, inputs, pos_inputs, training):
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        ### TODO(Students) START
        # ...
        word_pos = tf.concat([word_embed,pos_embed],axis=2)
        mask=tf.where(inputs>0,1.,0.)

        biGRU = self.bidir(word_pos,training=training,mask=mask)
        attention_embed = self.attn(biGRU)
        logits = self.decoder(attention_embed)

        ### TODO(Students) END

        return {'logits': logits}


class MyAdvancedModel(models.Model):

    def __init__(self,vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False,batch_size:int = 10):
        super(MyAdvancedModel, self).__init__()
        ### TODO(Students) START
        # ...
        self.num_classes = len(ID_TO_CLASS)
        self.hidden_size=hidden_size
        self.training=training
        self.vocab_size=vocab_size
        self.embed_dim=embed_dim
        self.batch_size = batch_size
        self.decoder = layers.Dense(units=self.num_classes)
        self.omegas = tf.Variable(tf.random.normal((hidden_size * 12, 1)))

        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))
        self.conv1layer = layers.Conv1D(filters=512, kernel_size=2, input_shape=(None, 2 * embed_dim), activation='relu');
        self.maxpool1 = layers.MaxPool1D(pool_size=2)

        self.conv2layer = layers.Conv1D(filters=1024, kernel_size=3, input_shape=(None, 2 * embed_dim), activation='relu');
        self.maxpool2 = layers.MaxPool1D(pool_size=2)


        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(hidden_size,input_shape=[self.batch_size,None])

        self.lstm_layer = layers.LSTM(6*self.hidden_size,return_sequences=True)

        self.bidir_lstm = layers.Bidirectional(self.lstm_layer, merge_mode='concat')
        ### TODO(Students END


    def conv_net(self,word_pos):
        sent_len= word_pos.shape[1]
        conv_output = self.conv1layer(word_pos)
        conv_output2 = self.conv2layer(word_pos)
        pool_output = layers.MaxPool1D(pool_size=sent_len-1)(conv_output)
        pool_output2 = layers.MaxPool1D(pool_size=sent_len-2)(conv_output2)
        pool_concat = tf.concat([pool_output,pool_output2],axis=2)
        flatten_output = self.flatten(pool_concat)
        return flatten_output

    def attention(self, word_pos):
        ### TODO(Students) START
        # ...
        M = tf.nn.tanh(word_pos)
        dot_M = tf.tensordot(M, self.omegas, axes=1)
        alpha = tf.nn.softmax(dot_M, axis=1)
        reshaped_alpha = alpha * tf.ones((1, 12 * self.hidden_size))
        r = tf.reduce_sum(word_pos * reshaped_alpha, axis=1)

        output = tf.nn.tanh(r)

        ### TODO(Students) END

        return output

    def call(self, inputs, pos_inputs):
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        # raise NotImplementedError
        ### TODO(Students) START
        # ...
        word_pos = tf.concat([word_embed, pos_embed], axis=2)
        mask=tf.where(inputs>0,1.,0.)

        CNN = self.conv_net(word_pos)
        lstm= self.bidir_lstm(tf.squeeze(word_embed),training=True,mask=mask)
        lstm_attn = self.attention(lstm)
        final_feat = tf.concat([CNN,lstm_attn],axis=1)
        logits = self.decoder(final_feat)

        return {'logits': logits}

        ### TODO(Students END
