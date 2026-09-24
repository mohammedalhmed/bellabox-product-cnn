# مشروع BellaBox Product Vision CNN

## الهدف

بناء نموذج **CNN لتصنيف فئات منتجات متجر بيلابوكس من صورة المنتج**. الاستخدام العملي هو توفير تصنيف بصري مبدئي عند إضافة منتج جديد إلى المتجر، ومساعدة فريق المتجر في اكتشاف الفئة المحتملة قبل المراجعة البشرية. النموذج لا ينفذ نشرًا تلقائيًا للمنتج ولا يستبدل مراجعة فريق المتجر.

## مصدر البيانات

يُبنى Dataset من المنتجات والصور العامة المنشورة في Sitemap متجر بيلابوكس: `https://bellaboxksa.com/sitemap.xml`. لا يستخدم المشروع Dataset تعليميًا جاهزًا. يقرأ الـ Sitemap صور المنتجات الأصلية من CDN الخاص بمتجر بيلابوكس، ويستخرج فئة التدريب من اسم المنتج/الرابط بقواعد قابلة للمراجعة. يمكن لاحقًا استبدال القواعد بعمود فئة مُراجع يدويًا في `manifest.csv` قبل التدريب.

## افتراضات واضحة

1. المطلوب نموذج رؤية حاسوبية قابل للتشغيل على Google Colab، وليس دمجه مباشرة داخل ثيم سلة.
2. المهمة الأولى هي **تصنيف الفئة** لا التعرف على SKU بعينه؛ التعرف على SKU يتطلب عدة صور مميزة لكل SKU وDataset أكبر بكثير.
3. صور المنتج المتعددة تُعامل كمجموعة واحدة في التقسيم؛ لا يجوز أن تظهر صورة من نفس المنتج في التدريب والاختبار.
4. لا تُرفع الصور أو الأوزان الكبيرة إلى GitHub؛ يتم تنزيلها وبناء النموذج داخل Google Drive/Colab.
5. التسميات المستخرجة آليًا هي نقطة بداية. يجب مراجعة `manifest.csv` وعينات الفئات قبل التسليم النهائي.

## التقنية

- Python 3 وTensorFlow/Keras.
- EfficientNetB0 كـ CNN backbone مع Transfer Learning من ImageNet، ثم Fine-tuning محدود.
- Augmentation للصور، Dropout، Label Smoothing، Class Weights، Early Stopping، ReduceLROnPlateau.
- `BackupAndRestore` و`ModelCheckpoint` لاستئناف التدريب وعدم فقدان التقدم في Colab.
- تقييم بـ Accuracy وMacro F1 وClassification Report وConfusion Matrix.

## الأوامر

```bash
# داخل Google Colab
!git clone https://github.com/mohammedalhmed/bellabox-product-cnn.git
%cd bellabox-product-cnn
!pip install -r ml/requirements-colab.txt
!python ml/build_dataset.py --output-dir /content/bellabox_dataset --min-images-per-class 10 --download
!python ml/train.py --data-dir /content/bellabox_dataset --output-dir /content/bellabox_outputs --epochs 15 --batch-size 32
!python ml/predict.py --model /content/bellabox_outputs/final_model.keras --labels /content/bellabox_outputs/labels.json --image /content/test-product.jpg
```

للتجربة السريعة دون تنزيل الصور:

```bash
python ml/build_dataset.py --output-dir /tmp/bellabox_dataset --no-download --max-products 100
```

## البنية

```text
ml/
├── build_dataset.py       # قراءة Sitemap، استخراج الصور، تنزيل الصور، وبناء manifest
├── train.py               # التقسيم، التدريب، التقييم، Checkpoints، وحفظ النموذج
├── predict.py             # اختبار صورة واحدة من سطر الأوامر
└── requirements-colab.txt
notebooks/
└── BellaBox_Product_CNN_Colab.ipynb
ML_PROJECT.md              # هذه المواصفات وخطوات التشغيل
```

## استراتيجية منع مشاكل التدريب

| المشكلة | المعالجة في المشروع |
|---|---|
| تسريب الصور بين المجموعات | تقسيم Grouped حسب `product_id` قبل تكوين Dataset |
| Overfitting | Augmentation، Dropout، Label Smoothing، تجميد الـ backbone أولًا، Fine-tuning محدود، Early Stopping |
| عدم توازن الفئات | حساب `class_weight` من مجموعة التدريب فقط |
| Underfitting | Transfer Learning، مرحلتان للتدريب، وفك آخر طبقات CNN بمعدل تعلم صغير |
| فقدان جلسة Colab | `BackupAndRestore` إلى مجلد ثابت في Google Drive عند استخدام Notebook |
| تلف أفضل وزن | `best.keras` و`last.keras` وCheckpoints لكل Epoch |
| تقييم مضلل بسبب تكرار المنتج | اختبار مستقل على منتجات لم تظهر في التدريب |

## معايير النجاح

- وجود Dataset حقيقي من Sitemap بيلابوكس، وليس Dataset تعليميًا جاهزًا.
- وجود فئتين على الأقل و10 صور على الأقل لكل فئة بعد التنظيف، مع إيقاف واضح إذا لم يتحقق الحد. يمكن رفع الحد بعد مراجعة Dataset.
- استكمال التدريب مع حفظ `best.keras` و`last.keras` و`final_model.keras` و`labels.json`.
- إنتاج `classification_report.txt` و`confusion_matrix.png` و`metrics.json`.
- نجاح اختبار صورة واحدة وإرجاع Top-3 احتمالات.
- إمكانية استئناف التدريب من مجلد Checkpoints في Colab.

## حدود المشروع

- **دائمًا:** تحقق من الروابط، تجاهل الصور التالفة، افصل المنتجات قبل التقسيم، وثّق مصدر البيانات وتاريخ البناء.
- **يتطلب مراجعة قبل اعتماده:** تغيير قواعد التسميات، خفض الحد الأدنى للصور، أو اعتماد النموذج للتصنيف التلقائي في المتجر.
- **ممنوع:** رفع مفاتيح أو بيانات دخول، رفع صور العملاء، أو نشر تصنيف آلي دون مراجعة بشرية.

## قرار التسليم

النسخة الحالية تسلّم Pipeline قابلًا لإعادة الإنتاج على Colab. دقة النموذج لا تُثبت إلا بعد تشغيل تنزيل البيانات والتدريب على حساب Colab؛ لذلك لا يتم اختلاق نسبة دقة مسبقًا. تُحفظ النتيجة الفعلية في `metrics.json` بعد التشغيل.

## مراجع تشغيلية

- [Sitemap متجر بيلابوكس](https://bellaboxksa.com/sitemap.xml)
- [مستودع المشروع المستقل](https://github.com/mohammedalhmed/bellabox-product-cnn)
- [Google Colab](https://colab.research.google.com/)
