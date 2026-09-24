# BellaBox Product CNN

مشروع تعلم آلي مستقل لتصنيف فئات منتجات متجر بيلابوكس من صور المنتجات باستخدام **CNN**. المشروع منفصل تمامًا عن ثيم سلة، ويعمل كـ Pipeline قابل لإعادة التشغيل على Google Colab من جمع البيانات إلى التدريب والتقييم واختبار صورة جديدة.

## ما الذي يفعله المشروع؟

يبني المشروع Dataset حقيقيًا من Sitemap متجر بيلابوكس العام وصور المنتجات المستضافة على CDN الخاص بالمتجر. يستخرج روابط المنتجات والصور، ينشئ تسميات فئات قابلة للمراجعة، ينزل الصور، ثم يدرب نموذج CNN لتوقع فئة المنتج. الهدف العملي هو إعطاء تصنيف بصري مبدئي عند إضافة منتج جديد، مع إبقاء المراجعة البشرية قبل اعتماد التصنيف.

المشروع لا يستخدم Datasets تعليمية جاهزة، ولا يرفع صور المنتجات أو أوزان النموذج إلى GitHub. يتم حفظ البيانات والنتائج داخل Google Drive أثناء تشغيل Google Colab.

## مستودع المشروع

`https://github.com/mohammedalhmed/bellabox-product-cnn`

## تشغيل Google Colab

يفضل فتح الملف [BellaBox_Product_CNN_Colab.ipynb](notebooks/BellaBox_Product_CNN_Colab.ipynb) في Google Colab، ثم تفعيل GPU من `Runtime > Change runtime type > T4 GPU` وربط Google Drive. الـ Notebook يثبت المتطلبات، ينزل بيانات المتجر، يدرب النموذج، يحفظ Checkpoints، ويختبر صورة جديدة.

يمكن أيضًا تشغيل الأوامر التالية داخل Colab بعد استنساخ المستودع:

```bash
!git clone https://github.com/mohammedalhmed/bellabox-product-cnn.git
%cd bellabox-product-cnn
!pip install -r ml/requirements-colab.txt
```

### 1. بناء Dataset

```bash
!python ml/build_dataset.py \
  --output-dir /content/bellabox_dataset \
  --min-images-per-class 10 \
  --max-images-per-product 3 \
  --download
```

سينتج الأمر `manifest.csv` و`dataset_summary.json` ومجلد `images/`. يجب مراجعة `manifest.csv` قبل التدريب للتأكد من صحة التسميات. الحد الأدنى الافتراضي هو 10 صور لكل فئة، ويمكن رفعه بعد مراجعة حجم البيانات.

### 2. تدريب CNN وحفظ النموذج

```bash
!python ml/train.py \
  --data-dir /content/bellabox_dataset \
  --output-dir /content/bellabox_outputs \
  --epochs 15 \
  --batch-size 32 \
  --resume
```

يستخدم التدريب EfficientNetB0 كـ CNN backbone مع Transfer Learning، Augmentation، Dropout، Label Smoothing، Class Weights، Early Stopping، ReduceLROnPlateau، وتقسيمًا Grouped حسب `product_id` لمنع تسريب صور المنتج نفسه بين التدريب والاختبار.

### 3. اختبار صورة جديدة

```bash
!python ml/predict.py \
  --model /content/bellabox_outputs/final_model.keras \
  --labels /content/bellabox_outputs/labels.json \
  --image /content/test-product.jpg \
  --top-k 3
```

سيظهر ناتج JSON يحتوي على أعلى ثلاثة تصنيفات واحتمال كل تصنيف.

## Checkpoints والنتائج

عند استخدام Notebook، اجعل `WORK_DIR` داخل Google Drive. يحفظ التدريب الملفات التالية داخل مجلد النتائج:

| الملف | الغرض |
|---|---|
| `best.keras` | أفضل أوزان بناءً على `val_macro_f1` |
| `last.keras` | آخر أوزان محفوظة |
| `final_model.keras` | النموذج النهائي بعد التقييم |
| `labels.json` | ترتيب الفئات المستخدم في النموذج |
| `metrics.json` | Accuracy وMacro Precision وMacro Recall وMacro F1 |
| `classification_report.txt` | تقرير تفصيلي لكل فئة |
| `confusion_matrix.png` | مصفوفة الالتباس |
| `training_curves.png` | منحنيات الخسارة والدقة |
| `backup_stage1/` و`backup_stage2/` | ملفات استئناف `BackupAndRestore` |

إذا انقطعت جلسة Colab، أعد تشغيل خلية التدريب مع نفس مجلد النتائج؛ سيحاول التدريب الاستئناف من ملفات Drive بدل البدء من الصفر.

## بنية المشروع

```text
.
├── ML_PROJECT.md
├── README.md
├── ml/
│   ├── build_dataset.py
│   ├── train.py
│   ├── predict.py
│   └── requirements-colab.txt
├── notebooks/
│   └── BellaBox_Product_CNN_Colab.ipynb
└── tasks/
    ├── plan.md
    └── todo.md
```

## التحقق المحلي

يمكن فحص صياغة Python وNotebook دون تثبيت TensorFlow:

```bash
python3 -m py_compile ml/build_dataset.py ml/train.py ml/predict.py
python3 -m json.tool notebooks/BellaBox_Product_CNN_Colab.ipynb >/dev/null
git diff --check
```

لا يتم تشغيل التدريب النهائي داخل مستودع GitHub أو على جهاز محلي؛ التدريب المقصود لهذا المشروع هو Google Colab مع GPU كما تنص متطلبات المشروع.

## ملاحظات مهمة للتسليم

البيانات الحالية تُجمع من Sitemap بيلابوكس وقت التشغيل، ولذلك يجب تسجيل محتوى `dataset_summary.json` ونتيجة `metrics.json` في يوم التدريب. لا تعتمد نسبة دقة قبل تشغيل التدريب الفعلي. كذلك يجب تجربة النموذج على صور منتجات لم تظهر في Dataset إن أمكن، ومراجعة النتائج يدويًا قبل اعتبار التصنيف جاهزًا للاستخدام.

## الترخيص

MIT. راجع [ML_PROJECT.md](ML_PROJECT.md) للمواصفات التفصيلية والافتراضات ومعايير النجاح.
