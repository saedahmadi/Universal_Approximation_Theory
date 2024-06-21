import tensorflow as tf
import numpy as np

# تعریف تابع f(x)
def f(x):
  return (x**3) + (x**2) - (x) - 1

# بررسی شکل ورودی و در صورت نیاز تغییر شکل آن
def prepare_input(x):
  # Reshape to (number of samples, 1) regardless of input shape
  x_reshaped = tf.reshape(x, [-1, 1])  
  return x_reshaped



# تعریف مدل شبکه عصبی
model = tf.keras.Sequential([
  tf.keras.layers.Dense(100, activation='relu', input_shape=(1,)),  # لایه اول با 10 گره و فعال سازی ReLU, note the input shape
  tf.keras.layers.Dense(1)  # لایه خروجی با 1 گره
])

# کامپایل مدل
model.compile(optimizer='adam', loss='mse')

# داده های آموزشی
x_train = np.linspace(-2, 2, 100)  # 100 نقطه داده در بازه [-2, 2]
y_train = f(x_train)

# تبدیل داده های آموزشی به شکل مناسب - Moved after x_train is defined
x_train_prepared = prepare_input(x_train) 
# y_train_prepared = f(x_train_prepared) # No need to recalculate y_train

# آموزش مدل - Use the prepared data
model.fit(x_train_prepared, y_train, epochs=1000)  # آموزش مدل برای 1000 دوره

# پیش بینی مقادیر برای ورودی های جدید
x_new = np.array([-1.5, 0.5, 1.8])
# Prepare the new input as well
x_new_prepared = prepare_input(x_new)  
y_pred = model.predict(x_new_prepared)

# نمایش نتایج
print("f(-1.5) = ", y_pred[0], "original f(x)= ", f(-1.5))
print("f(0.5)  = ", y_pred[1], "original f(x)= ",f(0.5))
print("f(1.8)  = ", y_pred[2], "original f(x)= ",f(1.8))
