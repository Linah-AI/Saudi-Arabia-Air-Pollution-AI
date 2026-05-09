import pandas as pd


def load_and_clean_data(file_path):

    # قراءة ملف الإكسل
    df = pd.read_excel(file_path)
    
    # معرفة الأسماء الحقيقية
    print("📌 أسماء الأعمدة في ملف الإكسل هي:")
    print(df.columns)

    # حذف القيم الفارغة
    df = df.dropna()

# حذف الصفوف التي تحتوي "لا يعاني"
    df = df[df["عانت من التلوث"] != "لا يعاني"]

    # دالة تصنيف مستوى الخطورة
    def classify_risk(value):

        if value >= 10:
            return "High"

        elif value >= 5:
            return "Medium"

        else:
            return "Low"

    # إنشاء عمود Risk بناءً على النسبة
    df["Risk"] = df["التوزيع النسبي"].apply(classify_risk)

    # إرجاع البيانات بعد التنظيف
    return df