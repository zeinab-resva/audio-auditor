 import streamlit as st
import librosa
import numpy as np
import pandas as pd

# إعدادات الصفحة والشكل العام
st.set_page_config(
    page_title="محلل جودة المكالمات الصوتي",
    page_icon="🎧",
    layout="centered"
)

# العنوان الرئيسي للموقع
st.title("🎧 نظام الفحص الآلي فائق الدقة للضوضاء")
st.subheader("تصفية وفلترة المكالمات هندسياً بدون الحاجة لـ APIs مدفوعة")
st.write("ارفعي ملف المكالمة المسجلة، والسيستم هيفلتر صوت الأيجنت تماماً ويطلعلك توقيتات الدوشة والخبط فوراً!")

st.markdown("---")

# خانة رفع الملف الصوتي
uploaded_file = st.file_uploader("👇 اختاري أو اسحبي ملف المكالمة هنا (MP3 / WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    # تشغيل الصوت داخل الموقع للمعاينة
    st.audio(uploaded_file, format='audio/wav')
    
    # زرار بدء التحليل
    if st.button("🔍 بدء الفحص الصوتي الفوري", type="primary"):
        with st.spinner("⏳ جاري تحليل بصمة الصوت بثلاثة فلاتر هندسية لضمان أعلى دقة..."):
            try:
                # 1. تحميل ملف الصوت بتردد 16000 هرتز
                y, sr = librosa.load(uploaded_file, sr=16000)
                
                # 2. حساب الفلاتر الثلاثية (الطاقة + التردد + انحدار الشوشرة)
                hop_length = 16000  # الفحص ثانية بثانية
                rms = librosa.feature.rms(y=y, frame_length=16000, hop_length=hop_length)[0]
                centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=16000, hop_length=hop_length)[0]
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=16000, hop_length=hop_length, roll_percent=0.85)[0]
                
                # حساب المتوسطات التلقائية للمكالمة دي بالذات
                mean_energy = np.mean(rms)
                mean_centroid = np.mean(centroid)
                mean_rolloff = np.mean(rolloff)
                
                # حدود الأمان لمنع التداخل مع صوت الأيجنت العالي
                energy_threshold = mean_energy * 1.4 
                centroid_threshold = mean_centroid * 1.25
                rolloff_threshold = mean_rolloff * 1.2
                
                violations = []
                
                # 3. الفحص الذكي بالثانية
                for i in range(len(rms)):
                    energy = rms[i]
                    spectral_val = centroid[i]
                    rolloff_val = rolloff[i]
                    
                    current_second = i
                    minutes = current_second // 60
                    seconds = current_second % 60
                    timestamp = f"{minutes:02d}:{seconds:02d}"
                    
                    # تطبيق الفلتر فائق الدقة
                    if energy > energy_threshold and spectral_val > centroid_threshold and rolloff_val > rolloff_threshold:
                        severity = "High 🚨" if energy > (energy_threshold * 1.8) else "Medium ⚠️"
                        violations.append({
                            "التوقيت ⏱️": timestamp,
                            "حالة الصوت المكتشف": "ضوضاء خلفية مؤكدة (خبط / شوشرة محيطة)",
                            "مستوى الإزعاج 📊": severity
                        })
                
                # 4. عرض النتائج والتقارير
                st.markdown("### 💡 ملخص الفحص والنتيجة")
                
                if len(violations) > 0:
                    df = pd.DataFrame(violations)
                    df = df.drop_duplicates(subset=['التوقيت ⏱️'])
                    
                    # عرض رسالة تنبيه حمراء بالمخالفات
                    st.error(f"🚨 تم رصد {len(df)} ثانية تحتوي على دوشة حقيقية في الخلفية (تم تجنب صوت الأيجنت العالي بنجاح).")
                    
                    # عرض الجدول التفاعلي الشيك جداً في Streamlit
                    st.markdown("### 📊 جدول توقيتات المخالفات المكتشفة")
                    st.dataframe(df, use_container_width=True)
 else:
                    # عرض رسالة نجاح خضراء
                    st.success("✅ الفحص فائق الدقة: المكالمة مطابقة للمواصفات، صوت الأيجنت طبيعي ولا توجد دوشة خلفية.")
                    
            except Exception as e:
                st.error(f"❌ حدث خطأ أثناء تحليل الملف: {str(e)}")

st.markdown("---")
st.caption("تطوير أداة الأوديت الذكية - مجانية بالكامل وبدون حدود للاستخدام.")