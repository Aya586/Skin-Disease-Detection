import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# تحميل الموديل
model = tf.keras.models.load_model(r"C:\CropMonitoringProject\Skindisease_Model.h5")

# تأكدي من ترتيب الكلاسات الأبجدي عشان النتائج تطلع صح
class_names = [
    'Acne', 
    'Melanocytic_Nevi_Moles', 
    'Melanoma', 
    'Normal_Skin', 
    'Seborrheic_Keratoses'
]

def predict_skin_disease(img_path):
    # تحميل ومعالجة الصورة
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # عمل التوقع
    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    confidence = prediction[0][index] * 100

    # عرض الصورة والتنسيق المطلوب
    plt.figure(figsize=(6, 8)) # تحديد حجم النافذة
    plt.imshow(img)
    plt.axis('off') # إخفاء المحاور
    
    # إضافة النص تحت الصورة بشكل منظم
    result_text = f"Prediction: {class_names[index]}\nAccuracy: {confidence:.2f}%"
    
    # وضع النص في العنوان أو أسفل الصورة
    plt.title(result_text, fontsize=14, color='blue', pad=20)
    
    plt.show()

# تجربة الدالة
predict_skin_disease(r"C:\Users\ayaab\Downloads\train\Train\Normal_Skin\beauty-woman-healthy-clean-skin-260nw-1938409180.jpg")