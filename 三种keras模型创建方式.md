三种构建模型的方法

# keras 序列模型

数据预处理

```python
mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x, test_x = tf.cast(train_x/255.0, dtype=float32), tf.cast(test_x/225.0, dtype=float32)

train_y, test_y = tf.cast(train_y, dtype=int64), tf.cast(test_y, dtype=int64)

```

序列模型

```python
model1 = tf.keras.modes.Sequential(
	[
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ]
)
```

模型编译

```python
optimizer = tf.keras.optimizers.Adam()
model1.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs)

model.evaluate(test_x, test_y)
```

## 方法二



# 方法三

```python
class MyModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        inputs = tf.keras.Input(shape=(28, 28))
        self.x0 = tf.keras.layers.Flatten()
        self.x1 = tf.keras.layers.Dense(512, activation='relu', name='d1')
        self.x2 = tf.keras.layers.Dropout(0.2)
        self.predictions = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name='d2')
    
    def call(self, inputs):
        x = self.x0(inputs)
        x = self.x1(x)
        x = self.x2(x)
        return self.predictions(x)
model4 = MyModel()
        
```

