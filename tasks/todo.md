# BellaBox ML Tasks

- [x] فحص ملف منتجات سلة Excel المرفق.
  - Acceptance: تأكيد وجود أعمدة المنتج والتصنيف والصور وفهم بنية القالب.
  - Verify: قراءة ورقة `Salla Products Template Sheet` وصف العناوين الثاني.

- [x] كتابة المواصفات وخطة التنفيذ.
  - Acceptance: توثيق الهدف، الافتراضات، البنية، الأوامر، المخاطر، ومعايير النجاح.
  - Verify: وجود `ML_PROJECT.md` و`tasks/plan.md`.

- [x] تنفيذ أداة بناء Dataset حقيقي من ملف منتجات Excel.
  - Acceptance: قراءة أعمدة سلة، استخراج labels من `تصنيف المنتج`، وقراءة روابط `صورة المنتج` مع إنشاء manifest.
  - Verify: تم تحليل الملف المرفق: 1252 صفًا، 419 منتجًا بصور، و789 رابط صورة قبل الحد الأقصى للصورة لكل منتج.

- [ ] تنفيذ تدريب CNN وتقييمه.
  - Acceptance: Group split، augmentation، class weights، checkpoints، وتقارير التقييم.
  - Verify: تشغيل التدريب في Colab ووجود ملفات `best.keras` و`metrics.json`.

- [ ] تنفيذ سكربت اختبار صورة واحدة.
  - Acceptance: إرجاع Top-3 احتمالات من النموذج المحفوظ.
  - Verify: تشغيل `python ml/predict.py ...` على صورة من Dataset.

- [x] إنشاء Notebook Google Colab وتحديث README.
  - Acceptance: خطوات تشغيل متسلسلة ومناسبة للتسليم.
  - Verify: نجح التحقق من JSON وMarkdown والأوامر والروابط الأساسية.

- [ ] تشغيل اختبارات محلية ودفع التغييرات إلى GitHub.
  - Acceptance: لا أخطاء syntax، وGitHub يحتوي الملفات الجديدة.
  - Verify: `python -m py_compile ml/*.py` و`git diff --check` ثم `git push`.

## سجل المراجعة

آخر مراجعة: 2026-09-24. تم اختبار قراءة وتحليل ملف Excel المرفق وفحص الأكواد. لم يتم تشغيل تنزيل كامل أو تدريب GPU داخل هذه البيئة؛ التشغيل النهائي يجب أن يتم على Google Colab كما طلبت متطلبات المشروع.
