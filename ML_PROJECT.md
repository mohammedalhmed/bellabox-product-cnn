# مشروع BellaBox Product Vision CNN

## الهدف

بناء نموذج **CNN لتصنيف فئات منتجات متجر بيلابوكس من صورة المنتج**. الاستخدام العملي هو توفير تصنيف بصري مبدئي عند إضافة منتج جديد إلى المتجر، ومساعدة فريق المتجر في اكتشاف الفئة المحتملة قبل المراجعة البشرية. النموذج لا ينفذ نشرًا تلقائيًا للمنتج ولا يستبدل مراجعة فريق المتجر.

## مصدر البيانات

المصدر الرسمي للـDataset هو ملف تصدير منتجات سلة بصيغة Excel، وليس Sitemap. يحتوي الملف على بيانات المنتج وعمود `تصنيف المنتج` وعمود `صورة المنتج`. يقرأ المشروع الصف الثاني كصف عناوين افتراضيًا، يستخرج الفئة من مسار سلة، وينزل روابط الصور من CDN. الملف المرفق وقت التطوير يحتوي على 1252 صف منتج و419 منتجًا يحتوي على روابط صور.

يتم حفظ نسخة الملف داخل Google Drive أثناء تشغيل Colab، ولا يتم رفعه إلى GitHub. كل صورة في `manifest.csv` مرتبطة بـ `product_id` حتى يتم تقسيم المنتجات كمجموعات ومنع تسريب صور المنتج نفسه بين التدريب والاختبار.

## افتراضات واضحة

1. المطلوب نموذج رؤية حاسوبية قابل للتشغيل على Google Colab، وليس دمجه مباشرة داخل ثيم سلة.
2. المهمة الأولى هي **تصنيف الفئة** لا التعرف على SKU بعينه؛ التعرف على SKU يتطلب Dataset مختلفًا بعدد صور أكبر لكل منتج.
3. `No.` هو معرف المنتج، و`تصنيف المنتج` هو label، و`صورة المنتج` يحتوي رابطًا أو عدة روابط مفصولة بفواصل.
4. المستوى الافتراضي للتصنيف هو 2، مثل `العناية > العناية بالوجه`. يمكن استخدام المستوى 1 للتصنيف العام أو المستوى 3 لتصنيف أدق مع بيانات كافية.
5. لا تُرفع صور المنتجات أو ملف Excel أو الأوزان إلى GitHub؛ يتم تنزيلها وبناء النموذج داخل Google Drive/Colab.
6. يجب مراجعة `manifest.csv` و`dataset_summary.json` قبل التدريب النهائي.

## التقنية

- Python 3 وTensorFlow/Keras.
- EfficientNetB0 كـ CNN backbone مع Transfer Learning من ImageNet، ثم Fine-tuning محدود.
- `openpyxl` لقراءة قالب Excel الخاص بسلة.
- Augmentation، Dropout، Label Smoothing، Class Weights، Early Stopping، ReduceLROnPlateau.
- `BackupAndRestore` و`ModelCheckpoint` لاستئناف التدريب وعدم فقدان التقدم في Colab.
- تقييم بـ Accuracy وMacro F1 وClassification Report وConfusion Matrix.

## الأوامر

```bash
# داخل Google Colab
!git clone https://github.com/mohammedalhmed/bellabox-product-cnn.git
%cd bellabox-product-cnn
!pip install -r ml/requirements-colab.txt
!python ml/build_dataset.py \
  --products-xlsx /content/drive/MyDrive/BellaBox_Product_CNN/bellabox_products.xlsx \
  --output-dir /content/drive/MyDrive/BellaBox_Product_CNN/dataset \
  --category-level 2 \
  --min-images-per-class 20 \
  --min-products-per-class 4 \
  --drop-small-classes \
  --download
!python ml/train.py \
  --data-dir /content/drive/MyDrive/BellaBox_Product_CNN/dataset \
  --output-dir /content/drive/MyDrive/BellaBox_Product_CNN/outputs \
  --epochs 15 --batch-size 32 --resume
!python ml/predict.py \
  --model /content/drive/MyDrive/BellaBox_Product_CNN/outputs/final_model.keras \
  --labels /content/drive/MyDrive/BellaBox_Product_CNN/outputs/labels.json \
  --image /content/test-product.jpg
```

للتجربة السريعة دون تنزيل الصور:

```bash
python ml/build_dataset.py \
  --products-xlsx /path/to/products.xlsx \
  --output-dir /tmp/bellabox_dataset \
  --category-level 2 --no-download --max-products 100
```

## البنية

```text
ml/
├── build_dataset.py       # قراءة Excel، استخراج الفئات والصور، وبناء manifest
├── train.py               # التقسيم، التدريب، التقييم، Checkpoints، وحفظ النموذج
├── predict.py             # اختبار صورة واحدة من سطر الأوامر
└── requirements-colab.txt
notebooks/
└── BellaBox_Product_CNN_Final_Colab.ipynb
```

## استراتيجية منع مشاكل التدريب

| المشكلة | المعالجة في المشروع |
|---|---|
| تسريب الصور بين المجموعات | تقسيم Grouped حسب `product_id` قبل تكوين Dataset |
| Overfitting | Augmentation، Dropout، Label Smoothing، تجميد الـ backbone أولًا، Fine-tuning محدود، Early Stopping |
| عدم توازن الفئات | حساب `class_weight` من مجموعة التدريب فقط |
| Underfitting | Transfer Learning، مرحلتان للتدريب، وفك آخر طبقات CNN بمعدل تعلم صغير |
| فئات صغيرة جدًا | حد أدنى للصور والمنتجات، مع خيار `--drop-small-classes` وتسجيل الفئات المستبعدة |
| فقدان جلسة Colab | `BackupAndRestore` إلى مجلد ثابت في Google Drive عند استخدام Notebook |
| تلف أفضل وزن | `best.keras` و`last.keras` وCheckpoints لكل Epoch |
| تقييم مضلل بسبب تكرار المنتج | اختبار مستقل على منتجات لم تظهر في التدريب |

## معايير النجاح

- بناء Dataset من ملف Excel الخاص بالمتجر، وليس Sitemap أو Dataset تعليميًا جاهزًا.
- قراءة أعمدة `No.` و`أسم المنتج` و`تصنيف المنتج` و`صورة المنتج` بنجاح.
- وجود فئتين على الأقل والحد الأدنى المحدد من الصور والمنتجات لكل فئة بعد التنظيف.
- استكمال التدريب مع حفظ `best.keras` و`last.keras` و`final_model.keras` و`labels.json`.
- إنتاج `classification_report.txt` و`confusion_matrix.png` و`metrics.json`.
- نجاح اختبار صورة واحدة وإرجاع Top-3 احتمالات.
- إمكانية استئناف التدريب من مجلد Checkpoints في Colab.

## حدود المشروع

- **دائمًا:** تحقق من أسماء الأعمدة، تجاهل الصور التالفة، افصل المنتجات قبل التقسيم، وثّق اسم ملف Excel وتاريخ البناء.
- **يتطلب مراجعة قبل اعتماده:** تغيير مستوى التصنيف، خفض الحدود الدنيا، أو اعتماد النموذج للتصنيف التلقائي في المتجر.
- **ممنوع:** رفع مفاتيح أو بيانات دخول، رفع ملف Excel أو صور العملاء، أو نشر تصنيف آلي دون مراجعة بشرية.

## قرار التسليم

النسخة الحالية تسلّم Pipeline قابلًا لإعادة الإنتاج على Colab. دقة النموذج لا تُثبت إلا بعد تشغيل تنزيل البيانات والتدريب على حساب Colab؛ لذلك لا يتم اختلاق نسبة دقة مسبقًا. تُحفظ النتيجة الفعلية في `metrics.json` بعد التشغيل.

## مراجع تشغيلية

- [مستودع المشروع المستقل](https://github.com/mohammedalhmed/bellabox-product-cnn)
- [Google Colab](https://colab.research.google.com/)
