# BellaBox Product CNN

مشروع تعلم آلي مستقل لتصنيف فئات منتجات متجر بيلابوكس من صور المنتجات باستخدام **CNN**. المشروع منفصل تمامًا عن ثيم سلة، ويعمل كـ Pipeline قابل لإعادة التشغيل على Google Colab من ملف منتجات سلة بصيغة Excel إلى التدريب والتقييم واختبار صورة جديدة.

## ما الذي يفعله المشروع؟

يستخدم المشروع **تصدير منتجات سلة Excel** كمصدر البيانات الأساسي. يقرأ عمود `تصنيف المنتج` كتسمية تدريب، ويقرأ روابط الصور من عمود `صورة المنتج`، ثم ينزل الصور إلى Google Drive وينشئ `manifest.csv`. بعد ذلك يدرب نموذج CNN على فئات المتجر الفعلية، مع إبقاء المراجعة البشرية قبل اعتماد التصنيف.

المشروع لا يستخدم Sitemap ولا Dataset تعليميًا جاهزًا، ولا يرفع ملف Excel أو صور المنتجات أو أوزان النموذج إلى GitHub. البيانات والنتائج تبقى داخل Google Drive أثناء تشغيل Google Colab.

## مستودع المشروع

`https://github.com/mohammedalhmed/bellabox-product-cnn`

## شكل ملف Excel المطلوب

يدعم المشروع قالب منتجات سلة الذي يحتوي على صف عنوان في الصف الثاني، وبالأخص الأعمدة التالية:

| العمود | الاستخدام |
|---|---|
| `No.` | معرف المنتج وتجميع الصور لمنع تسريب المنتج بين المجموعات |
| `أسم المنتج` | اسم المنتج للتوثيق |
| `تصنيف المنتج` | الفئة التي تتحول إلى label |
| `صورة المنتج` | رابط صورة أو عدة روابط مفصولة بفواصل |

في الملف الحالي يوجد 1252 صف منتج، و419 منتجًا يحتوي على روابط صور، وملفات الصور تأتي من CDN سلة. عدد الصفوف الفعلي قد يتغير إذا تم تصدير ملف جديد من لوحة سلة.

## تشغيل Google Colab

افتح [Notebook BellaBox_Product_CNN_Colab.ipynb](notebooks/BellaBox_Product_CNN_Colab.ipynb) في Google Colab، فعّل GPU من `Runtime > Change runtime type > T4 GPU`، واربط Google Drive. ستطلب الخلية الثالثة رفع ملف Excel مرة واحدة، ثم تحفظه في مجلد Drive.

يمكن أيضًا تشغيل الأوامر يدويًا بعد استنساخ المستودع:

```bash
!git clone https://github.com/mohammedalhmed/bellabox-product-cnn.git
%cd bellabox-product-cnn
!pip install -r ml/requirements-colab.txt
```

### 1. بناء Dataset من Excel

```bash
!python ml/build_dataset.py \
  --products-xlsx /content/drive/MyDrive/BellaBox_Product_CNN/bellabox_products.xlsx \
  --output-dir /content/drive/MyDrive/BellaBox_Product_CNN/dataset \
  --category-level 2 \
  --min-images-per-class 20 \
  --min-products-per-class 4 \
  --drop-small-classes \
  --max-images-per-product 3 \
  --download
```

`--category-level 2` يحول المسار إلى تصنيف متوسط مثل `العناية > العناية بالوجه` أو `المكياج > العيون`. استخدم `--category-level 1` لتصنيف عام مثل `العناية` و`المكياج`. المستوى 3 أكثر تفصيلًا لكنه يحتاج بيانات أكثر لكل فئة.

ينتج الأمر `manifest.csv` و`dataset_summary.json` ومجلد `images/`. يجب مراجعة `manifest.csv` قبل التدريب. الفئات التي تقل عن الحد الأدنى يتم استبعادها فقط عند استخدام `--drop-small-classes`، ويسجل الاستبعاد في `dataset_summary.json`.

### 2. تدريب CNN وحفظ النموذج

```bash
!python ml/train.py \
  --data-dir /content/drive/MyDrive/BellaBox_Product_CNN/dataset \
  --output-dir /content/drive/MyDrive/BellaBox_Product_CNN/outputs \
  --epochs 15 \
  --batch-size 32 \
  --resume
```

يستخدم التدريب EfficientNetB0 كـ CNN backbone مع Transfer Learning، Augmentation، Dropout، Label Smoothing، Class Weights، Early Stopping، ReduceLROnPlateau، وتقسيمًا Grouped حسب `product_id` لمنع تسريب صور المنتج نفسه بين التدريب والاختبار.

### 3. اختبار صورة جديدة

```bash
!python ml/predict.py \
  --model /content/drive/MyDrive/BellaBox_Product_CNN/outputs/final_model.keras \
  --labels /content/drive/MyDrive/BellaBox_Product_CNN/outputs/labels.json \
  --image /content/test-product.jpg \
  --top-k 3
```

سيظهر ناتج JSON يحتوي على أعلى ثلاثة تصنيفات واحتمال كل تصنيف.

## Checkpoints والنتائج

عند استخدام Notebook، احفظ `WORK_DIR` داخل Google Drive. يحفظ التدريب الملفات التالية داخل مجلد النتائج:

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

يمكن فحص صياغة Python وNotebook دون تشغيل التدريب:

```bash
python3 -m py_compile ml/build_dataset.py ml/train.py ml/predict.py
python3 -m json.tool notebooks/BellaBox_Product_CNN_Colab.ipynb >/dev/null
git diff --check
```

لا يتم تشغيل التدريب النهائي داخل مستودع GitHub أو على جهاز محلي؛ التدريب المقصود لهذا المشروع هو Google Colab مع GPU كما تنص متطلبات المشروع.

## ملاحظات مهمة للتسليم

سجل محتوى `dataset_summary.json` ونتيجة `metrics.json` في يوم التدريب. لا تعتمد نسبة دقة قبل تشغيل التدريب الفعلي. جرّب النموذج على صور منتجات لم تظهر في Dataset إن أمكن، وراجع الفئات في `manifest.csv` يدويًا قبل اعتبار التصنيف جاهزًا للاستخدام.

## الترخيص

MIT. راجع [ML_PROJECT.md](ML_PROJECT.md) للمواصفات التفصيلية والافتراضات ومعايير النجاح.
