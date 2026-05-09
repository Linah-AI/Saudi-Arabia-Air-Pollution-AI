import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from src import preprocessing, training, evaluation

def main():
    print("🚀 بدء تشغيل نظام التنبؤ بتلوث الهواء في مكة...")
    print("-" * 50)

    # 1. مرحلة تنظيف البيانات (شغل سلام)
    # ---------------------------------
    data_path = "data/makkah_pollution.xlsx"
    clean_df = preprocessing.load_and_clean_data(data_path)
    
    print("✅ تم تنظيف البيانات بنجاح!")
    print("\n📊 عينة من الجداول اللي نظفتها سلام (أول 5 صفوف):")
    print(clean_df.head())
    print("-" * 50)

    # 2. مرحلة تدريب النموذج (شغل دانة)
    # ---------------------------------
    # هنا المودل بيذاكر البيانات عشان يتعلم القواعد
    model, X_test, y_test = training.train_model(clean_df)
    
    print("✅ تم تدريب المودل (الذكاء الاصطناعي) بنجاح!")
    print(f"🧠 المودل الآن خبير وعنده {model.get_n_leaves()} قاعدة استنتاجية.")
    print("-" * 50)

    # 3. مرحلة الاستعراض البصري (عشان تشوفون الشجرة)
    # ------------------------------------------
    print("🌳 جاري رسم شجرة القرار (Decision Tree)...")
    plt.figure(figsize=(20, 10))
    plot_tree(model, 
              feature_names=["Year", "Pollution Type"], # أسماء المدخلات
              class_names=model.classes_,               # مستويات الخطورة (Low, Med, High)
              filled=True, 
              rounded=True, 
              fontsize=10)
    plt.title("Makkah Pollution Decision Tree Model")
    print("💡 ستفتح الآن نافذة جديدة تحتوي على رسمة الشجرة.. أغلقيها للاستمرار.")
    plt.show()

    # 4. مرحلة التقييم (شغل خديجة)
    # ----------------------------
    # بمجرد ما تخلص خديجة ملف evaluation.py، هذا السطر بيطلع لكم نسبة الدقة
    try:
        evaluation.evaluate_performance(model, X_test, y_test)
    except AttributeError:
        print("\n⚠️ تنبيه: كود التقييم (خديجة) لسه مو جاهز أو ناقص.. بانتظار خديجة!")

    print("\n" + "="*50)
    print("🎉 انتهت جميع العمليات بنجاح! كفو يا فريق مكة!")
    print("="*50)

if __name__ == "__main__":
    main()
