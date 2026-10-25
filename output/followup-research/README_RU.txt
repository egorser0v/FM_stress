ДОПИССЛЕДОВАНИЕ FM STRESS — 25 ОКТЯБРЯ 2026

Исходная версия: 0333803e2563b38b98073b4ed2009c41e3889bfd, изменения Василия Лямина.
Новые коммиты и push не создавались. Результаты предыдущих исследований сохранены.

ГЛАВНЫЕ РЕЗУЛЬТАТЫ

Выполнены 144 новых MPS-обучения: 36 контролей архитектуры/бюджета,
84 варианта нормализации/источника и 24 теста условно усреднённой цели.
Это 1 080 000 шагов оптимизации. Два бюджета одной траектории не считаются
независимыми запусками. Дополнительно 60 моделей проверены на полном ансамбле
при Euler 64 и 128 без переобучения.

1. Числа Василия воспроизводятся: 360 старых checkpoints на всех 8192 парах,
   плюс полные прогнозные ансамбли 18 целевых ячеек. Уточняется интерпретация.
2. На RBF/F64 с белым шумом исходный MLP ширины 56 сохраняет шум минимум
   в восьми выходных направлениях. Контроль почти равного размера убирает
   79.4% разрыва остаточной ошибки с S4; более широкий — 88.4%.
   Шероховатость падает с 2.68 нормы target до 1.06/1.00.
   При GP-источнике эффект расширения пренебрежимо мал. Это расширение
   длинного горизонта, а не доказательство исходной гипотезы на F16.
3. На ETTh1/GP CRPS MLP/S4 меняется с 0.2165/0.1748 при global scaling
   до 0.0982/0.0981 при одном центрировании. Порядок голов зависит от
   нормализации. Удаление постоянной компоненты GP не спасает global MLP.
4. Усреднённая условная цель уменьшает residual discrepancy во всех 12 парах.
   CRPS улучшается в 6/6 пар MLP и 4/6 S4; смена Euler64 на128 не меняет
   знаки сравнений. Этот teacher использует известное синтетическое
   распределение, поэтому это проверка механизма, не готовый real-data метод.
5. Якобианный вывод зависит от информации эталона и весов направлений.
   Для GP unweighted ошибка ниже у S4, а covariance-weighted и конечные
   локальные изменения — у MLP. Это не прямое сравнение качества прогнозов.
6. Сходный CRPS скрывает различия в ширине и покрытии интервалов.
   В ETTh1 102 из118 тестовых историй ниже тренировочного Q25 по изменчивости;
   в высокой группе всего одна история. Малые отклонения покрытия от90%
   нельзя автоматически считать дефектом: конечный ансамбль даёт смещение.

Все основные независимые проверки прошли, включая 211 тестов проекта.
Буквальные критерии исходного задания и прежние pass/fail не переписаны.
На новых 24 label-моделях все проходят roughness, ни одна — полное
velocity-ratio правило. Универсальное превосходство S4 или MLP не установлено.

ЧТО ЧИТАТЬ

Итоговый связный отчёт: output/pdf/followup-research.pdf.
Версия для браузера: output/followup-research/followup-research.html.
Файлы с суффиксом interim — промежуточные снимки; они не являются итогом.
План, границы исследования и порядок фиксации настроек:
docs/followup_research_plan.md.

ПАКЕТЫ ДОПИССЛЕДОВАНИЯ

1. Независимое воспроизведение результатов Василия.
   docs/followup_reproduction_audit.md
   results/followup-reproduction/all_summary.json
   output/followup-reproduction/decoder_null_variance.png

2. Аналитический эталон с той же информацией, которая доступна нейросети.
   docs/followup_oracle_protocol.md
   output/followup-oracle/report.html
   results/followup-oracle/independent_verification.json

3. Ширина MLP, ограничение выходного пространства и бюджет обучения.
   docs/followup_capacity_protocol.md
   results/followup-capacity/execution_notes.txt
   output/followup-capacity/summary.json и runs.csv

4. Центрирование, масштабирование и постоянная компонента GP на ETTh1.
   docs/followup_normalization_protocol.md
   output/followup-normalization/summary.json и runs.csv
   results/followup-normalization-audit/summary.json и analysis.json

5. Обычная цель flow matching против её точного условного среднего.
   docs/followup_label_noise_protocol.md
   results/followup-label-noise/execution_notes.txt
   output/followup-label-noise/report.html и summary.json
   results/followup-label-noise/stream_verification.json

6. Конечные возмущения поля вместо сравнения одних производных.
   output/followup-oracle/finite_perturbation/report.html
   results/followup-oracle/finite_perturbation/independent_arithmetic_audit.json

7. Проверка калибровки по изменчивости наблюдаемой истории.
   docs/followup_calibration_strata_protocol.md
   results/followup-calibration-strata/summary.json
   output/followup-calibration-strata/strata.csv

8. Численная устойчивость прогнозов при Euler64 и Euler128.
   docs/followup_solver_protocol.md
   results/followup-solver/summary.json
   Все 36 моделей длинного горизонта, без переобучения и смены checkpoint.
   Дополнительно все 24 модели эксперимента с усреднённой целью:
   output/followup-label-solver/report.html
   results/followup-label-solver/cross_review_verification.json

В обучающих пакетах сохранены настройки, seed, контрольные суммы, контрольные
точки моделей, прогнозы и метрики отдельных запусков. Аналитические проверки и
повторное интегрирование используют существующие checkpoints. Нельзя считать два бюджета одной
траектории обучения двумя независимыми экспериментами.

КАК ПОВТОРИТЬ

Все команды выполняются из корня проекта в его Python-окружении.
Обучение требует доступного Apple MPS; автоматической подмены на CPU нет.
Точные зависимости указаны в requirements-lock.txt, фактические версии —
в results/followup-capacity/environment.json,
results/followup-normalization/environment.json и
results/followup-label-noise/environment_mps.json.

Обучение (завершённые совместимые результаты пропускаются):
  .venv/bin/python -m fm_stress.followup_capacity --device mps
  .venv/bin/python -m fm_stress.followup_normalization --device mps
  .venv/bin/python -m fm_stress.followup_label_noise --prepare-only
  .venv/bin/python -m fm_stress.followup_label_noise --device mps

Не запускайте два процесса на одинаковых ячейках: результаты имеют общие имена.
Для новых настроек задайте отдельный --output. Манифест защищает уже сохранённое
исследование от незаметного смешивания кода и конфигураций.

Агрегация завершённых результатов:
  .venv/bin/python scripts/analyze_followup_capacity.py
  .venv/bin/python scripts/analyze_followup_normalization.py
  .venv/bin/python scripts/analyze_followup_label_noise.py

Проверка поставляемых готовых артефактов:
  .venv/bin/python scripts/audit_followup_reproduction.py --phase all
  .venv/bin/python scripts/check_normalized_history_oracle.py
  .venv/bin/python scripts/check_followup_capacity.py
  .venv/bin/python scripts/check_followup_normalization.py
  .venv/bin/python scripts/check_followup_normalization_analysis.py
  .venv/bin/python scripts/check_followup_label_noise.py
  .venv/bin/python scripts/check_followup_label_noise_hashes.py
  .venv/bin/python scripts/analyze_followup_label_solver.py
  .venv/bin/python scripts/check_followup_label_solver_independent.py
  .venv/bin/python scripts/report_matched_support.py
  .venv/bin/python scripts/analyze_followup_calibration_strata.py
  .venv/bin/python scripts/audit_followup_solver.py --device cpu
  .venv/bin/python scripts/check_followup_solver_independent.py
  .venv/bin/python -m pytest -q

Для восстановления дополнительных входов отчёта, если они удалены, нужны
исходные checkpoints и данные из репозитория. Выполните в таком порядке:
  .venv/bin/python scripts/audit_followup_reproduction.py --phase all
  .venv/bin/python scripts/audit_followup_decoder_span.py
  .venv/bin/python scripts/plot_followup_decoder_span.py
  .venv/bin/python scripts/study_normalized_history_oracle.py
  .venv/bin/python scripts/check_normalized_history_oracle.py
  .venv/bin/python scripts/report_normalized_history_oracle.py
  .venv/bin/python scripts/probe_matched_support.py
  .venv/bin/python scripts/report_matched_support.py
  .venv/bin/python scripts/audit_followup_scope_geometry.py
  .venv/bin/python scripts/audit_followup_no_flow.py
  .venv/bin/python scripts/check_followup_label_noise_hashes.py
  .venv/bin/python scripts/study_followup_label_solver.py
  .venv/bin/python scripts/analyze_followup_label_solver.py
  .venv/bin/python scripts/check_followup_label_solver_independent.py
  .venv/bin/python scripts/audit_followup_solver.py --prepare
  .venv/bin/python scripts/audit_followup_solver.py --device cpu
  .venv/bin/python scripts/check_followup_solver_independent.py
  .venv/bin/python scripts/analyze_followup_calibration_strata.py --prepare
  .venv/bin/python scripts/analyze_followup_calibration_strata.py

Эти команды дополнительно к обучению и проверкам выше. Для калибровочных групп
сначала нужны полная агрегация нормализации и её независимый аудит. Флаг prepare
записывает фиксированные настройки; при повторении сегодня он не означает,
что новые решения были приняты до просмотра исторических результатов.

Сборка итогового отчёта после завершения проверок:
  .venv/bin/python scripts/plot_followup_paths.py
  .venv/bin/python scripts/build_controlled_research_report.py

ВАЖНО ДЛЯ ИНТЕРПРЕТАЦИИ

Здесь отдельно рассматриваются ошибка скорости, локальная производная поля,
шероховатость прогнозов и вероятностное качество прогнозирования. Улучшение
одного из этих показателей не означает автоматического улучшения остальных.
Три синтетические выборки и один ранее изученный период ETTh1 не доказывают
универсального превосходства архитектуры. Новые эксперименты не изменяют
первоначальный критерий успеха и не подменяют результаты исходного задания.
