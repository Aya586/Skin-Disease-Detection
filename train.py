import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# 1. إعداد البيانات
# -----------------------------
TRAIN_PATH = r"C:\Users\ayaab\Downloads\train\Train"
VAL_PATH   = r"C:\Users\ayaab\Downloads\train\Validate"
TEST_PATH  = r"C:\Users\ayaab\Downloads\train\Test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Train (فيه augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Validation & Test (بدون augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    VAL_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_gen = val_datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("Classes:", train_gen.class_indices)

# -----------------------------
# 2. بناء الموديل (أفضل شوية)
# -----------------------------
model = models.Sequential([
    tf.keras.Input(shape=(224,224,3)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation='relu'),   # قللناها من 256
    layers.Dropout(0.5),

    layers.Dense(train_gen.num_classes, activation='softmax')
])

# -----------------------------
# 3. Compile
# -----------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# 4. Early Stopping
# -----------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# -----------------------------
# 5. Training
# -----------------------------
print("\n🚀 Training Started...\n")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[early_stop]
)

# -----------------------------
# 6. Test Evaluation
# -----------------------------
print("\n📊 Evaluating on Test Data...\n")
test_loss, test_acc = model.evaluate(test_gen)

print(f"Test Accuracy: {test_acc * 100:.2f}%")

# -----------------------------
# 7. حفظ الموديل
# -----------------------------
model.save('Skindisease_Model.h5')
print("\n✅ Model Saved Successfully!")