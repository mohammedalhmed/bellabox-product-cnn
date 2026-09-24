# دليل التسليم النهائي لمشروع BellaBox Product CNN

## الهدف

يوفر هذا الدليل طريقة تشغيل منظمة للنموذج على Google Colab باستخدام GPU، مع حفظ Dataset والنموذج والتقارير وCheckpoints داخل Google Drive. ملف المصدر هو تصدير منتجات سلة Excel، وليس Sitemap أو Dataset تعليميًا جاهزًا.

## بنية مجلد Google Drive

```text
BellaBox_Product_CNN_Colab_Project/
├── BellaBox_Product_CNN_Colab_Project.zip
├── bellabox_products.xlsx
├── dataset/
│   ├── manifest.csv
│   ├── dataset_summary.json
│   └── images/
├── outputs_final/
│   ├── best.keras
│   ├── last.keras
│   ├── final_model.keras
│   ├── labels.json
│   ├── metrics.json
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── training_curves.png
│   ├── training_log.csv
│   ├── split_summary.json
│   ├── run_config.json
│   ├── backup_stage1/
│   └── backup_stage2/
└── BellaBox_CNN_Final_Run.zip
```

## Notebook النهائي

استخدم `notebooks/BellaBox_Product_CNN_Final_Colab.ipynb`. تم تقسيمه إلى مراحل مستقلة:

1. ربط Google Drive وفك ضغط المشروع عند الحاجة.
2. تثبيت المتطلبات وفحص TensorFlow وGPU.
3. فحص وجود `dataset/manifest.csv` والصور المحلية.
4. تخطي إعادة البناء إذا كانت Dataset موجودة وجاهزة.
5. بناء Dataset من Excel فقط إذا لم تكن موجودة.
6. فحص تقسيم المنتجات والتأكد من عدم وجود Product Leakage.
7. بدء التدريب أو استئنافه من `backup_stage1` و`backup_stage2`.
8. التحقق من النموذج النهائي وملفات التقارير.
9. إنشاء تقرير تصنيف وConfusion Matrix مُطبّعة.
10. اختبار صورة جديدة اختياريًا وأرشفة نتائج التشغيل.

## شرط تجاوز بناء Dataset

إذا وجدت الخلية `manifest.csv`، ووجدت صورًا محلية ذات `download_status=ok`، واحتوت على فئتين على الأقل، فستظهر الرسالة:

```text
✅ Dataset للتدريب والاختبار موجودة في المسار المحدد.
لن تتم إعادة بناء Dataset أو تنزيل الصور.
```

لا تعدّل `dataset/manifest.csv` أو تحذف مجلد `dataset/images/` بعد ذلك، لأن التدريب يعتمد على المسارات المحلية المسجلة في manifest.

## شرط استئناف التدريب

استخدم دائمًا نفس:

```text
OUTPUT_DIR=/content/drive/MyDrive/BellaBox_Product_CNN_Colab_Project/outputs_final
```

ولا تحذف هذا المجلد عند انقطاع جلسة Colab. أعد تشغيل خلية التدريب نفسها مع `--resume`. يحفظ التدريب:

- `best.keras` عند تحسن `val_macro_f1`.
- `last.keras` في كل Epoch.
- `backup_stage1/` لاستئناف المرحلة الأولى.
- `backup_stage2/` لاستئناف Fine-tuning.
- `training_log.csv` لتتبع Epochs.

إذا كان `final_model.keras` موجودًا، يتخطى Notebook التدريب تلقائيًا ويكمل مرحلة التحقق. للتدريب الجديد، استخدم مجلد نتائج جديدًا مثل `outputs_run_02` بدل حذف النتائج القديمة.

## التقييم النهائي

يجب عرض:

- Accuracy.
- Macro Precision.
- Macro Recall.
- Macro F1.
- Classification Report لكل فئة.
- Confusion Matrix بالأعداد.
- Confusion Matrix مُطبّعة بالنسبة لكل فئة.
- منحنيات التدريب والتحقق.
- اختبار صورة جديدة لم تدخل في Dataset.

لا تعتمد Accuracy وحدها لأن أعداد الصور تختلف بين الفئات. في ملف Excel الحالي توجد فئات صغيرة، ولذلك يجب تفسير نتائجها بحذر وذكر قيمة `support` في التقرير.

## قائمة فحص التسليم

- [ ] تم تفعيل GPU في Colab.
- [ ] ظهرت نسخة TensorFlow المتوافقة.
- [ ] ظهرت رسالة Dataset الموجودة أو اكتمل بناؤها من Excel.
- [ ] تم التأكد من عدم تقاطع `product_id` بين train وvalidation وtest.
- [ ] تم حفظ `best.keras` و`last.keras` و`final_model.keras`.
- [ ] تم حفظ `metrics.json` و`classification_report.txt`.
- [ ] تم عرض Confusion Matrix العادية والمطبّعة.
- [ ] تم اختبار صورة جديدة خارج Dataset.
- [ ] تم ضغط `outputs_final/` في `BellaBox_CNN_Final_Run.zip`.
- [ ] تم حفظ نسخة ZIP النهائية في Google Drive.

## القيود التي يجب ذكرها

النموذج مصنف بصري مبدئي لفئات المنتجات، وليس نظام نشر تلقائي داخل المتجر. الفئات الصغيرة تحتاج منتجات وصورًا إضافية حتى يصبح تقييمها أكثر استقرارًا. يجب اعتماد النتيجة بعد مراجعة بشرية، خصوصًا عند انخفاض الثقة أو عند الخلط بين فئات متشابهة بصريًا.
