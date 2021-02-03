import tensorflow as tf
import tensorflow_addons as tfa


def sent_char_encoder(x,vocab_dim,char_embed_size,word_embed_size):
    x = tf.keras.layers.Embedding(vocab_dim,char_embed_size,name='char_embeds')(x)
    x = tf.keras.layers.Conv1D(word_embed_size,3,activation='relu',use_bias=False)(x)
    x = tf.math.reduce_max(x,axis=2)
    x = tf.keras.layers.Dense(word_embed_size)(x)
    return(x)



def model(vocab_dim,char_embed_size,word_embed_size,use_char_level_embeddings=True, word_vectors=None,
                lstm_hidden=8,num_classes=10,dropout=0.3,use_pos_features=False,num_pos=10):
    
    mask = tf.keras.Input(shape=(None,),name='attention_masks',dtype=tf.bool)
    pos = tf.keras.Input(shape=(None,),name='pos_tags')
    if use_char_level_embeddings:
        sents = tf.keras.Input(shape=(None,15),name='input_ids')
        x = sent_char_encoder(sents,vocab_dim,char_embed_size,word_embed_size)
    else:
        sents = tf.keras.Input(shape=(None,))
        if word_vectors is none:
            word_initializer = tf.keras.initializers.Constant(word_vectors)
        else:
            word_initializer = tf.keras.initializers.GlorotNormal()
        
        x = tf.keras.layers.Embedding(vocab_dim,100,name='word_embeds',
                                            embeddings_initializer=word_initializer,
                                            trainable=False,)(sents)

    if use_pos_features:            
        x_pos = tf.keras.layers.Embedding(num_pos,num_pos,embeddings_initializer=tf.keras.initializers.Identity(),
                                      trainable=False,dtype=tf.float32,
                                      name='pos_embeddings')(pos)
        x = tf.concat([x,x_pos],axis=-1)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_hidden,return_sequences=True,dropout=dropout),
                                      name='lstm_enc')(x,mask=mask)
    # x = tf.keras.layers.Dense(128,activation='relu')(x)
    crf_layer = tfa.layers.CRF(num_classes,name='crf')
    a,b,_,_ = crf_layer(x)
    model = tf.keras.Model(inputs=[sents,mask,pos], outputs=[a,b], name="test")
    return(model)