# خطة تنفيذ BellaBox Product Vision CNN

## 1. مصدر البيانات والـ manifest

- قراءة ملف تصدير منتجات سلة Excel من Google Drive، مع استخدام صف العناوين الثاني.
- استخراج `product_id` من `No.`، واسم المنتج من `أسم المنتج`، والتصنيف من `تصنيف المنتج`، وروابط الصور من `صورة المنتج`.
- إنشاء label من مسار التصنيف مع دعم `--category-level`، بحيث يكون المستوى 2 هو الافتراضي.
- تنزيل الصور مع retry وUser-Agent، والتحقق من MIME وفتح الصورة، ثم حفظ `manifest.csv`.
- استبعاد الفئات الصغيرة اختياريًا عبر `--drop-small-classes` مع تسجيلها في `dataset_summary.json`.

## 2. إعداد التدريب

- قراءة `manifest.csv` وإزالة الصفوف التي لا تملك ملفات صالحة.
- تقسيم Grouped Stratified حسب `product_id` إلى train/validation/test؛ منع تكرار SKU بين المجموعات.
- إنشاء `tf.data` مع augmentation على التدريب فقط وclass weights محسوبة من train فقط.

## 3. نموذج CNN

- EfficientNetB0 بوزن ImageNet، رأس تصنيف مخصص بعد GlobalAveragePooling وDropout.
- المرحلة الأولى: تجميد backbone وتدريب الرأس.
- المرحلة الثانية: فك آخر طبقات backbone فقط مع معدل تعلم منخفض.
- حفظ `best.keras` عند تحسن `val_macro_f1`، و`last.keras` لكل Epoch، وBackupAndRestore للاستئناف.

## 4. التقييم والاختبار

- حساب Accuracy وMacro Precision/Recall/F1.
- إنتاج Classification Report وConfusion Matrix وملف metrics.
- توفير `predict.py` لإرجاع Top-3 احتمالات لصورة واحدة.

## 5. Google Colab

- Notebook يثبت المتطلبات، يربط Google Drive اختياريًا، يبني Dataset، يدرب النموذج، ويجرب صورة.
- مسار Checkpoints الافتراضي على Drive عند توفره لتجاوز انقطاع جلسة Colab.

## المخاطر والتخفيف

- **رفض روابط الصور:** retries، تأخير بين الطلبات، وتسجيل الصور الفاشلة بدل إيقاف كل العملية.
- **تسمية ضعيفة:** تقرير توزيع الفئات، ملف manifest قابل للتعديل، وعدم قبول فئات صغيرة.
- **قلة الصور:** حد أدنى صريح ورسالة توقف بدل تدريب غير موثوق.
- **تضخم الذاكرة:** `tf.data` streaming وcache اختياري فقط، وعدم تحميل كل الصور في RAM.
- **عدم اتزان الفئات:** class weights وMacro F1 بدل Accuracy فقط.

## نقاط التحقق

- بعد بناء manifest: فحص عدد المنتجات والصور وتوزيع الفئات.
- بعد التقسيم: التحقق من عدم وجود تقاطع product IDs.
- بعد التدريب: وجود الأوزان والتقارير والـ labels.
- بعد الاختبار: نجاح Top-3 وتسجيل حالة النموذج.
