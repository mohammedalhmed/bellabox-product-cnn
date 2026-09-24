# BellaBox ML Tasks

- [x] فحص مستودع الثيم وموقع بيلابوكس العام.
  - Acceptance: تأكيد وجود Sitemap للمنتجات وعدم وجود Dataset محلي.
  - Verify: `find . -type f` وقراءة `sitemap-2.xml`.

- [x] كتابة المواصفات وخطة التنفيذ.
  - Acceptance: توثيق الهدف، الافتراضات، البنية، الأوامر، المخاطر، ومعايير النجاح.
  - Verify: وجود `ML_PROJECT.md` و`tasks/plan.md`.

- [x] تنفيذ أداة بناء Dataset حقيقي من Sitemap.
  - Acceptance: إنشاء manifest وصور محلية مع تسمية قابلة للمراجعة والتحقق من الحد الأدنى.
  - Verify: نجح التشغيل على Sitemap الكامل: 721 منتجًا قابلًا للقراءة، 595 صورة، و7 فئات؛ ونجح تنزيل عينة صور حقيقية من CDN.

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

آخر مراجعة: 2026-09-24. تم اختبار بناء manifest وتنزيل صور حقيقية وفحص الأكواد. لم يتم تشغيل تدريب GPU داخل هذه البيئة؛ التشغيل النهائي يجب أن يتم على Google Colab كما طلبت متطلبات المشروع.
