from keras.models import load_model
model = load_model('../deep_font_top.h5')
print(model.summary())
print(model.layers[0].get_weights())
