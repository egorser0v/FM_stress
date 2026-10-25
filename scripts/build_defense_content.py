#!/usr/bin/env python3
"""Build slide content from the same evidence as the paper; no hand-copied scores."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
E=json.loads((ROOT/'output/final-paper/evidence.json').read_text())
T=json.loads((ROOT/'results/trajectory-metrics/summary.json').read_text())
P=json.loads((ROOT/'results/final-pca-audit/summary.json').read_text())
G={(g['suite'],g['group']):g for g in T['groups']}
slides=[]
def add(title,kind='text',seconds=60,backup=False,**kw):
 slides.append(dict(title=title,kind=kind,seconds=seconds,backup=backup,**kw))
def f(x,n=4):return f'{x:.{n}f}'
names={'mlp':'MLP','s4':'S4','whitened_mlp':'Whitened MLP'}
add('Smooth source,\nunstructured head',kind='cover',seconds=20,subtitle='Does a smooth starting curve change which forecasting model works better?',body=['Daniil Koblov · Vasilii Lyamin · Egor Serov · Kirill Frolov','Sequential Data project · 26 October 2026'],notes='20 секунд. Мы исследуем вероятностный прогноз временного ряда: не одно будущее, а несколько возможных траекторий. Вопрос — помогает ли гладкий начальный шум и нужна ли для него специальная последовательностная модель.')
add('The task: predict several plausible futures',seconds=55,body=['Input: the previous 32 observations of a time series.','Output: a distribution of the next 16 observations.','Controlled data: smooth random curves from a Gaussian process (GP).','Question: does a GP starting curve favour a sequence model over a dense MLP?'],bottom='Synthetic data let us compare learned forecasts with an analytic reference.',notes='55 секунд. Будущее не определяется прошлым однозначно. Поэтому выдаём много возможных будущих рядов. GP — распределение случайных кривых: соседние точки коррелируют. На синтетических данных мы знаем процесс генерации и можем построить эталон. Это небольшой контролируемый опыт, а не соревнование больших готовых forecasting систем. Источник: docs/protocol.md; docs/related_work.md.')
add('Flow matching learns how to move a random curve',seconds=65,body=['Start with a random future-length curve: independent noise or smooth GP noise.','During training, mix that curve with a real future at a random interpolation time.','Train a network to predict the direction of movement, given the past.','At prediction time, take small steps along the learned direction. Repeat for an ensemble.'],bottom='Same interpolation and solver for both heads. “Matched GP” matches the raw generator kernel.',notes='65 секунд. Представьте случайную кривую и наблюдённое будущее. На обучении выбираем точку между ними и учим скорость, которая ведёт от первой ко второй. При прогнозировании настоящего будущего у модели нет: она начинает с шума и многократно применяет выученное правило, учитывая историю. Гладкий источник использует тот же kernel, что сырые данные. После нормализации истории маргинальные распределения уже не обязаны совпадать. Источник: docs/protocol.md; Lipman et al., https://arxiv.org/abs/2210.02747.')
add('A controlled comparison with stated limitations',kind='table',seconds=60,headers=['Component','Choice'],rows=[['Heads','Dense MLP / temporal S4 / whitened MLP control'],['Main experiment','2 sources × 2 heads, plus GP whitening; 5 seeds each'],['Training','6,000 steps; matched encoder initialization; ≈65k parameters'],['Data split','32,768 train / 4,096 validation / 8,192 test windows'],['Leakage control','History-only normalization; train-only PCA; validation selects weights']],bottom='Conditioning differs between heads. Equal budgets do not imply equally tuned models.',notes='60 секунд. MLP смешивает весь прогноз плотными слоями. S4 обрабатывает временную структуру через модель состояния. Whitening меняет масштабы направлений данных и функцию потерь. Пять seed — это пять обучений, а не пять новых наборов данных. Преобразования оцениваются только на обучающей выборке или доступной истории. Исходный тест не выбирает чекпоинт. Последующий анализ уже exploratory. В MLP AdaLN, в S4 добавление контекста как bias: идеальной изоляции архитектуры нет. Источник: evidence.main.config; docs/protocol.md; docs/response_to_review.md.')
add('We measure different properties separately',kind='table',seconds=70,headers=['Measure','What it asks'],rows=[['Velocity MSE','Does the network predict the training direction accurately?'],['R1 / R2 ratios','Does generated movement / change of slope match the data?'],['CRPS','Are predicted values at each time calibrated and accurate?'],['Variogram score (VS)','Do predicted changes between times match observed changes?']],bottom='Lower MSE / CRPS / VS is better. R1 and R2 near 1 match a shape statistic only.',notes='70 секунд. Ошибка скорости относится к обучению, а не напрямую к качеству будущих рядов. PCA разбивает ошибку по направлениям изменения кривой: первое самое вариативное, остаточные — менее вариативные. R1 — средний модуль соседних разностей. R2 — средний модуль изменения наклона. CRPS оценивает распределение значений по времени. VS оценивает предсказанные попарные изменения и чувствителен к временной зависимости. Он не видит общий сдвиг кривой, поэтому нужен вместе с CRPS. Fair означает поправку на конечное число независимых прогнозов, не доказательство честного сравнения моделей. Источник: docs/trajectory_metrics.md; Scheuerer & Hamill 2015 https://repository.library.noaa.gov/view/noaa/22327/.')
rows=[]
for h in names:
 m=E['main']['groups']['gp|'+h]['metrics'];rows.append([names[h],f(m['pc1_mse']['mean'],3),f(m['residual_mse']['mean'],4)])
add('At horizon 16, MLP has lower velocity error than S4',kind='table',seconds=55,headers=['GP-source head','Leading-direction MSE','Residual-direction MSE'],rows=rows,bottom='Five-seed means. This is a field-regression result, not proof of better forecasts.',notes='55 секунд. При гладком источнике обычный MLP не проигрывает S4 по исходной ошибке скорости. Whitening уменьшает остаточную ошибку ещё сильнее. Формальный критерий задания требовал почти равных ошибок в первом и остаточных направлениях. Ни один из 25 запусков его не прошёл, потому что остаточная ошибка намного МЕНЬШЕ. Мы сохраняем этот отрицательный результат и не заменяем критерий после эксперимента. Источник: evidence.main.groups. Остаточная ошибка усреднена по PC3–16; PC2 отдельно в отчёте.')
add('The jump metric misses changes in slope',kind='chart',seconds=70,chart={'categories':[names[h] for h in names],'series':[{'name':'R1: movement ratio','values':[G['main',h]['means']['R1_ratio'] for h in names]},{'name':'R2: slope-change ratio','values':[G['main',h]['means']['R2_ratio'] for h in names]}],'axis':'Generated / observed roughness','format':'0.00'},bottom='R1 is near 1 for every head. S4 changes slope more strongly than the observed futures.',notes='70 секунд. Для монотонной кривой сумма модулей jumps зависит только от разницы между концами. Кривая может идти ровно или часто менять наклон, а R1 будет одинаковой. R2 ловит это различие: у S4 около 1.63, у двух MLP около 0.99. Но высокий R2 реагирует и на осмысленную кривизну. Мы не объявляем его новым критерием провала. Здесь все числа на одном новом ансамблевом наборе: 256 историй, 64 прогноза, Euler128, пять моделей на head. Источник: results/trajectory-metrics/summary.json main groups.')
rows=[[names[h],f(G['main',h]['means']['fair_crps']),f(G['main',h]['means']['variogram_fair'])] for h in names]
add('Extra curvature does not establish worse forecasts',kind='table',seconds=65,headers=['Head','Fair CRPS ↓','Fair variogram score ↓'],rows=rows,bottom='S4 vs MLP remains unresolved. Whitening lowers VS in all 5 paired seeds.',notes='65 секунд. Для S4 VS даже немного ниже, хотя R2 выше. Интервал разницы S4 минус MLP включает ноль: от −0.004371 до +0.000020. Поэтому нельзя сказать, что более неровные траектории хуже как вероятностный прогноз. Whitening улучшает VS во всех пяти seed; условный bootstrap-интервал не включает ноль. Но CRPS-различия всё ещё не разрешены. Интервалы условны на этих моделях и одной тестовой выборке, и метрики добавлены после просмотра результатов. Источник: results/trajectory-metrics/summary.json main contrasts.')
capnames={'mlp_original':'Original MLP','mlp_rank_clear':'Wider, same size','mlp_wide':'Larger MLP','s4':'S4'}
rows=[[name,f(G['followup-capacity',h]['means']['R2_ratio'],2),f(G['followup-capacity',h]['means']['variogram_fair']),f(G['followup-capacity',h]['means']['fair_crps'])] for h,name in capnames.items()]
add('At horizon 64, decoder width explains much of the gap',kind='table',seconds=75,headers=['White-source head','R2 ratio','Fair VS ↓','Fair CRPS ↓'],rows=rows,bottom='A 56-wide output layer cannot adjust source variance in every direction of a 64-point future.',notes='75 секунд. На длинном горизонте исходный MLP сжимает скрытое представление до 56 чисел перед выходом из 64 точек. Поэтому часть направлений исходного шума нельзя выученным движением очистить от случайной вариации. Контроль ширины65 с меньшей глубиной имеет примерно столько же параметров, как исходный. Он резко снижает R2 и VS. Но CRPS меняется мало. Значит, прежний выигрыш S4 на этом режиме частично объяснялся устройством выхода MLP. Это другой режим: RBF, белый источник, F64, 3 независимых набора с парными seed, бюджет12k, ансамбль32. На F16 ограничения56<64 нет. Источник: evidence.capacity; trajectory-metrics followup-capacity.')
norm=E['normalization']['table']
rows=[[name,*[f(norm[f'{mode}|gp|{h}|6000']['mean']['fair_crps']) for h in ('mlp','s4')]] for mode,name in [('global','Global scaling'),('center','Subtract history mean'),('history','History mean + scale')]]
add('On ETTh1, centering removes the large head gap',kind='table',seconds=65,headers=['Preprocessing','MLP CRPS ↓','S4 CRPS ↓'],rows=rows,bottom='Same 118 test windows. This real-data follow-up is exploratory and covers one period.',notes='65 секунд. ETTh1 — реальный ряд температуры трансформатора. Сравниваем одни и те же118 окон. При глобальном scaling S4 выглядит заметно лучше. Одно вычитание среднего доступной истории почти убирает этот разрыв. При полном scaling точность похожа, но покрытие интервалов хуже, поэтому одинаковый CRPS не означает одинаковой калибровки. Простая ridge-регрессия остаётся сильным baseline. Это один временной период, не широкое доказательство на реальных данных. Источник: evidence.normalization.table; docs/followup_normalization_final_review.md.')
add('Review feedback changed the interpretation',kind='table',seconds=60,headers=['Concern','What we checked / what remains'],rows=[['Tiny PCA directions','PC3–6 still rank whitening, MLP, S4 by error'],['Oracle knows more than model','Added an oracle with the same normalized history'],['One synthetic data draw','Capacity and label-noise controls use 3 paired draws'],['Short training / unfair conditioning','Longer budgets help; separate tuning and matched conditioning remain open']],bottom='We do not claim that one architecture is generally better.',notes='60 секунд. Ревью содержательное, мы не отмахиваемся от него. Проверка PC3–6 показывает, что порядок ошибок не создан только nugget-направлениями. Новый эталон получает ту же информацию, что сеть. Аналитически усреднённые обучающие цели уменьшают ошибку поля, но дают модели знание распределения и не изолируют только шум. Полного поиска learning rate, AdaLN-варианта S4, sweeps nugget и whitening floor нет. Мы прямо ограничиваем выводы. Числа и подробные ответы есть в резервных слайдах и таблице ответов. Источник: docs/response_to_review.md; results/final-pca-audit/summary.json.')
add('Supported conclusions',seconds=45,body=['A smooth GP source did not make the tested MLP worse than S4 in velocity regression at horizon 16.','R1 can hide large changes in slope. R2 diagnoses them, but does not by itself rank forecast quality.','Output width and history centering explain specific apparent architecture advantages.','Whitening improves some diagnostics. Its stability and broader forecasting benefit remain limited conclusions.'],bottom='Code, archived predictions and scores are released. AI assistance is disclosed in the report.',notes='45 секунд. Завершаем конкретными выводами, без универсального победителя. Самый важный урок: форма траектории, точность поля и качество распределения — разные свойства, нужны разные проверки. Наши изменения воспроизводятся по сохранённым прогнозам без нового обучения. Использовались AI-инструменты для разработки, проверки расчётов и подготовки материалов. До сдачи нужны человеческая проверка и репетиция.')
# Reserve slides: never included in the timed talk.
add('Backup: the literal success rule',seconds=0,backup=True,body=['Original rule: residual MSE / leading-PC MSE must lie in [0.8, 1.2].','Generated R1 / target R1 must also lie in [0.8, 1.2].','All 25 runs pass the roughness part; 0 / 25 pass the full rule.','Even the privileged analytic oracle fails the error-equality part.'],bottom='Keep the declared result. Diagnose why equality between unequal-variance directions is uninformative.',notes='PCA basis fitted on training velocities. Residual averages PC3–F, not their sum. Very small residual error should not normally count against accuracy. Noise-normalized scores remain floor-sensitive. Original rule is preserved, all new metrics explicitly post-hoc. Source: docs/protocol.md; output/final-paper/evidence.json.')
add('Backup: remove nugget-dominated PCs from the comparison',kind='table',seconds=0,backup=True,headers=['Head','Original MSE on PC3–6','Same-info oracle gap on PC3–6'],rows=[[names[h],f(P['main'][h]['raw_mse']['pc3_6'],6),f(P['fresh_matched_oracle'][h]['oracle_discrepancy']['pc3_6'],6)] for h in names],bottom='Different cohorts in the two columns. PC7 is a transition direction, not purely numerical noise.',notes='Original main MSE uses8192 heldout windows. Oracle comparison uses2048 fresh contexts. Direct oracle discrepancy is squared(model field−same-information oracle field), not naive subtraction of two finite-sample MSEs. The latter has a finite-sample cross term. Source: results/final-pca-audit/summary.json.')
add('Backup: averaged training labels improve forecasts modestly',kind='table',seconds=0,backup=True,headers=['GP-source model','Fair CRPS ↓','R2 ratio','Fair VS ↓'],rows=[[h.replace('_',' '),f(G['followup-label-noise',h]['means']['fair_crps']),f(G['followup-label-noise',h]['means']['R2_ratio'],3),f(G['followup-label-noise',h]['means']['variogram_fair'])] for h in ('mlp_stochastic','mlp_oracle','s4_stochastic','s4_oracle')],bottom='Same training inputs, 3 paired draws. Analytic labels supply privileged knowledge of the distribution.',notes='Oracle labels use conditional average of stochastic velocity label with the same model inputs. This is not the forecast oracle. Model checkpoint selection is based on validation field discrepancy for both label modes. 6000steps, GPsource, F16,32members. Results do not show a consistent proper-score advantage of S4 over MLP. Source: docs/followup_label_noise_protocol.md; trajectory-metrics summary.')
add('Backup: uncertainty and score definitions',seconds=0,backup=True,body=['VS uses all time pairs, exponent 0.5 and inverse-lag weights normalized to sum to 1.','Fair VS subtracts sample variance of the ensemble moment divided by ensemble size.','Main CIs resample 256 paired histories, conditional on 5 fixed models.','Follow-ups report 3 paired draw/seed differences. No pooled population CI.'],bottom='Post-hoc comparisons, no multiple-testing correction. Scores across different horizons are not directly comparable.',notes='For eachpair z=abs(Xi−Xj)^p, a=abs(yi−yj)^p. fairVS=sum w[(a−mean z)^2−var(z,ddof1)/M]. It can be negative, do not clip. Assumes iidmembers conditional on model/history. Variogram misses common shifts and is not strictlyproper. Source: docs/trajectory_metrics.md. FairCRPS and energy match original percontext numbers for all39runs.')
add('Backup: limitations',seconds=0,backup=True,body=['Separate learning-rate tuning, identical head conditioning and nugget/floor sweeps remain untested.','Main seeds share one data draw. Real-data evidence covers one ETTh1 period.','Most follow-up conclusions were motivated by earlier test results.','Fixed budgets and differing history conditioning limit architecture claims.'],bottom='AI assisted code, numerical checks and drafting. Do not infer individual scientific ownership from commit counts.',notes='See the paper for detailed limitations and the package README for reproduction and the frozen revision.')
# Revise for a first-time audience after the independent novice review.
slides[1].update(seconds=65, body=[
 'Given 32 past observations, generate several possible 16-point futures.',
 'We test on smooth random curves, so the data-generating process is known.',
 'Hypothesis: a sequence model may learn small local changes better when a smooth source makes broad trends dominate.',
 'Test: compare a dense network (MLP) with a temporal state-space network (S4).'])
slides[1]['notes']='65 секунд. Показываем не одно будущее, а несколько возможных. GP — способ генерировать плавные случайные кривые. Исходная гипотеза: при гладком стартовом шуме большая часть вариации — широкие изменения уровня и тренда. Маленькие локальные изменения может быть труднее выучить плотной сети. Возможно, временная структура S4 поможет. Это предположение, а не математическая гарантия. Проверяем его сравнением на одинаковых данных. Источник: docs/related_work.md.'
slides[2].update(body=[
 'Network input: the 32-point past, a current 16-point curve and interpolation time.',
 'Network output: 16 simultaneous updates, one for each future coordinate.',
 'Training teaches updates along paths from random noise to observed futures.',
 'Sampling repeats the learned update. A new random start gives another possible future.'],
 bottom='“Velocity” means an update to the whole forecast curve, not its slope through calendar time.')
slides[2]['notes']='65 секунд. Сеть не идёт по прогнозу слева направо: она меняет сразу все16координат. Интерполяционное время — ход генерации от шума к прогнозу, а не время самого ряда. На обучении правильная поправка известна, потому что есть стартовый шум и будущее. При прогнозировании будущее недоступно, сеть применяет выученное правило по истории. Разные случайные старты дают разные будущие. Источник: docs/protocol.md; Lipman et al. https://arxiv.org/abs/2210.02747.'
slides[3].update(seconds=55, rows=[
 ['MLP','Dense layers mix all future coordinates'],
 ['S4','State-space layers explicitly model temporal structure'],
 ['Whitened MLP','Same MLP type, with data directions rescaled before learning'],
 ['Comparison','White or smooth starting noise; 5 training seeds per setting'],
 ['Leakage control','Past-only normalization; training-only transforms; separate validation']],
 bottom='Both heads receive the same past, but use different conditioning mechanisms.')
slides[3]['notes']='55 секунд. Head — блок, который выдаёт обновления кривой. Модели получают одинаковые входные данные, но обрабатывают контекст по-разному, поэтому это не идеальная изоляция временной структуры. Whitening усиливает малые направления вариации при обучении: меняется масштаб координат и вклад ошибок в loss, а не добавляется новая последовательностная архитектура. Два источника на две модели плюс whitening-контроль, по5обучений —25запусков. Размеры выборок и точные настройки в отчёте. Источник: docs/protocol.md.'
slides[4]['headers']=['Measure','Question answered']
slides[4]['rows'][0]=['Training error (MSE)','How accurately does the model predict the update?']
slides[4]['bottom']='Forecast quality: use CRPS and VS together. R1/R2 are shape diagnostics, not a success test.'
slides[4]['notes']='70 секунд. Для качества будущих прогнозов главные здесь CRPS и VS: первое оценивает значения, второе — изменения между временами. Обе меньше — лучше; сравниваем только внутри одного эксперимента, абсолютный масштаб сам по себе не имеет порога хорошего прогноза. Ошибка скорости проверяет обучение. R1 — сколько кривая меняется между точками, R2 — как меняется наклон. Делим на соответствующий показатель данных:1означает совпадение среднего. Равенство одной характеристики не означает хорошее распределение. Источник: docs/trajectory_metrics.md.'
slides[5].update(title='The original error-equality rule rejects all 25 runs',seconds=70,
 bottom='Required: residual / leading error in [0.8, 1.2]. Here residual error is much smaller. GP-source means shown.')
slides[5]['notes']='70 секунд. В задании успех требовал почти одинаковой ошибки в самом изменчивом направлении формы и в остальных. В таблицеGPисточник: направление общего движения имеет намного больше вариации, поэтому его сырая ошибка намного больше. Все25моделей формально проваливают правило, хотя R1вдопуске. Даже аналитическийэталон его не проходит. Это показывает проблему критерия, а не доказывает плохой прогноз. Правило сохраняем, затем отдельно оцениваем прогнозы. В GPсравнении MLP имеет меньшую остаточную ошибку, чемS4;whitening ещё меньше. Не сравниваем абсолютные ошибки между направлениями как одинаковые задачи. Источник: evidence.main.groups; docs/protocol.md.'
slides[8].update(title='Wider MLP output greatly reduces excess roughness',
 bottom='Separate check: horizon 64, white source, 3 new datasets. Forecast CRPS changes only slightly.')
slides[8]['rows'][1][0]='Wider MLP, same ≈params'
slides[8]['notes']='75 секунд. Теперь отдельный контроль: 64 точки вместо 16, белый источник. В прежней серии S4 выглядел лучше. Оказалось, последняя скрытая ширина MLP равна 56: 64 выходных числа зависят лишь от 56 независимых комбинаций. Это мешает убрать исходную вариацию шума в некоторых направлениях. Увеличиваем ширину до 65 и уменьшаем глубину, сохраняя примерно число параметров. R2 падает с 19 до 1.69, VS тоже улучшается, но CRPS почти не меняется. На трёх новых наборах эффект повторяется. Это объясняет избыток неровности в конкретной конфигурации, а не доказывает универсальную победу S4 или MLP. Источник: evidence.capacity; results/trajectory-metrics/summary.json.'
slides[11].update(seconds=50,body=[
 'At horizon 16, the smooth source did not give S4 the expected advantage in training error.',
 'S4 has more slope changes, but its forecast scores do not establish worse quality than MLP.',
 'At horizon 64, widening the MLP output removes much of its excess roughness.',
 'Whitening lowers VS in this sample. Broader benefit remains uncertain; tuning and conditioning still differ.'],
 bottom='Review responses, real-data results and limitations follow in backup. AI assistance is disclosed.')
slides[5]['headers']=['Source / head','Leading-direction MSE','Residual-direction MSE']
slides[5]['rows']=[]
for source in ('white','gp'):
 for h in ('mlp','s4','whitened_mlp'):
  if source=='white' and h=='whitened_mlp':continue
  m=E['main']['groups'][source+'|'+h]['metrics']
  slides[5]['rows'].append([source.upper()+' / '+names[h],f(m['pc1_mse']['mean'],3),f(m['residual_mse']['mean'],4)])
slides[5]['bottom']='Required residual/leading error: 0.8–1.2. None of the 25 individual runs pass. Five-seed means shown.'
slides[5]['notes']+=' Белый источник тоже в таблице: по средней остаточной ошибке MLP не хуже S4 при обоих источниках. Сравнивайте модели внутри каждого источника: ошибки между источниками имеют разные шумовые уровни. Проверено каждое из25обучений отдельно, не только показанные средние.'
slides[7]['bottom']='GP source: S4 vs MLP is unresolved. Whitened MLP lowers VS in all 5 paired seeds.'
slides[8]['notes']+=' Увеличенный MLP ширины72 сохраняет глубину3; у варианта ширины65 глубина2. Два контроля согласуются с механизмом ограничения выхода, но не изолируют все последствия изменения архитектуры.'
slides[11]['notes']+=' По просьбе рецензентов отдельно исключили почти шумовые направления: порядок остаточных ошибок сохранился. Новые проверки не устранили ограничения разного способа подачи истории и неполного подбора настроек. Подробные ответы на каждый пункт в резерве.'
slides[4]['rows'][0]=['Update-prediction MSE','How accurately does the model predict held-out updates?']
slides[4]['rows'][2]=['CRPS','How accurate are the predictive values at each time?']
slides[11]['body'][0]=slides[11]['body'][0].replace('training error','held-out update error')
slides[11]['body'][1]=slides[11]['body'][1].replace('more slope changes','stronger changes in slope')
slides[8]['bottom']='Separate check: RBF, horizon 64, white source, 3 new datasets. CRPS changes only slightly.'
ridge=E['normalization_baselines']['history']['forecast_means']['fair_crps']
slides[9]['bottom']=f'GP source, 118 windows. History-normalized ridge CRPS: {ridge:.4f}. One previously examined period.'
slides[13]['headers'][0]='GP-source head'
slides[16]['body'][0]=slides[16]['body'][0].replace('Separate learning-rate tuning','Broad per-head learning-rate tuning')
# Reserve detailed real-data and review material for questions, preserving all content.
main=slides[:9]+[slides[11]]
extra=[slides[9],slides[10]]
for x in extra:x['backup']=True;x['seconds']=0;x['title']='Backup: '+x['title'][0].lower()+x['title'][1:]
slides=main+extra+slides[12:]
# Put actual trajectories and uncertainty in the timed talk, using archived data.
def figure_slide(index, filename, alt, **updates):
 s=slides[index]
 for key in ('body','headers','rows','chart'):s.pop(key,None)
 s.update(kind='figure',figure='output/final-presentation/figures/'+filename,alt=alt,**updates)

figure_slide(1,'forecast_task.png','Observed history, held-out future and four generated futures for the first test context.',
 bottom='32 observed points → 16 future points. Each generated curve is one possible future.')
slides[1]['notes']+=' На рисунке первая сохранённая история, четыре прогноза GP+MLP и одно фактически реализованное будущее. Они не обязаны совпадать: качество распределения проверяем на множестве историй. Этот пример иллюстративный, без отбора по метрикам. Источник и индекс окна указаны в figures/provenance.json.'
figure_slide(2,'source_curves.png','Independent white starting noise and correlated Gaussian-process starting noise.',
 title='Same forecasting method, two kinds of starting noise',
 figure_height=270,
 body=['Past + starting curve → learned updates to all future points → forecast',
       'Repeat with a new random start to obtain another possible future.'],
 bottom='Generation steps move the whole curve; they are not the time axis of the forecast.')
slides[2]['notes']+=' На графике показан только выбор стартового распределения, не этапы генерации. Это воспроизведение иллюстрации из prior-mixture: оба источника имеют единичную дисперсию координат. Масштаб иллюстрации не надо принимать за точную ковариацию основного обучения. Обучение и движение от старта к прогнозу объясняем схемой под графиком.'
figure_slide(7,'score_differences.png','Paired forecast-score differences from MLP, with conditional 95 percent bootstrap intervals.',
 bottom='Zero = no difference from MLP. S4 intervals cross zero; whitening lowers VS in this sample.')
slides[7]['notes']+=' Точка — разница средней метрики относительно MLP. Отрицательная разница лучше MLP; отрезок показывает условный 95%-й bootstrap-интервал. Если он пересекает ноль, эти результаты не разрешают порядок моделей. Интервалы не охватывают новые обучения или новые генераторы данных.'
figure_slide(8,'capacity_paths.png','First held-out context: original MLP, wider MLP and S4, each with four matched source draws.',
 bottom='RBF, white noise, horizon 64. Mean R2 across 3 datasets:\nOriginal MLP '+f(G['followup-capacity','mlp_original']['means']['R2_ratio'],2)+
 ' → wider MLP '+f(G['followup-capacity','mlp_wide']['means']['R2_ratio'],2)+
 '; S4 '+f(G['followup-capacity','s4']['means']['R2_ratio'],2)+'.',
 notes='75 секунд. Здесь отдельный эксперимент: RBF, белый источник, 64 будущих точки, 12 тысяч шагов обучения. На рисунке первая сохранённая история и первые четыре согласованных старта. Чёрная линия — одно реализованное будущее, цветные — возможные прогнозы; совпадение с чёрной линией не является критерием успеха отдельного прогноза. Слева MLP с последней скрытой шириной56 при выходе64: часть случайной вариации источника нельзя скорректировать. В центре увеличиваем ширину до72, сохраняя глубину3: видим гладкие траектории. Справа S4. Средний R2 по трём новым наборам падает с19.09 до1.60; уS4 около1.00. VS также улучшается, CRPS меняется мало. Дополнительный контроль ширины65 и глубины2 с близким числом параметров даёт тот же качественный результат; он есть в отчёте. Это подтверждает механизм ограничения выхода в данной конфигурации, но не универсальное превосходство архитектуры. Источники: figures/provenance.json и results/trajectory-metrics/summary.json.')
# The real-data backup is also easier to read as a comparison plot.
slides[10].update(kind='chart',chart={
 'categories':['Global scaling','History centering','History mean + scale'],
 'series':[{'name':names[h],'values':[norm[f'{mode}|gp|{h}|6000']['mean']['fair_crps'] for mode in ('global','center','history')]} for h in ('mlp','s4')],
 'axis':'Fair CRPS (lower is better)','format':'0.000'})
slides[10].pop('headers');slides[10].pop('rows')
# Required main-talk material: normalization, vector-valued forecasts and review.
multi=E['multivariate']
mc={(r['dataset'],r['source'],r['head']):r for r in multi['cells']}
datasets=['etth1_uni','etth1_multi','exchange_multi']
multislide=dict(title='Whitening worsens both real multichannel tasks',kind='chart',seconds=75,backup=False,
 chart={'categories':['ETTh1 OT (1 variable)','ETTh1 (7 variables)','Exchange (8 variables)'],
 'caption':'GP source · 2,000 updates · fixed training-channel scaling · 64 test histories',
 'series':[{'name':names[h],'values':[mc[d,'gp',h]['metrics']['fair_crps']['mean'] for d in datasets]} for h in names],
 'axis':'Fair CRPS (lower is better)','format':'0.000'},
 bottom='Compare heads within each task, not CRPS across tasks. Channel count is not isolated.\nRidge + residuals scores lower than all three heads on each task.',
 notes='75 секунд. Теперь в каждый момент прогнозируем вектор величин, а не только один ряд. ETTh1 OT — только температура, ETTh1 целиком — семь показателей, Exchange — восемь валютных рядов. Сети совместно используют и прогнозируют каналы. Сравнивайте столбики только внутри каждой задачи, а не абсолютные значения между задачами. В этой отдельной серии GP-источника whitening улучшает OT: CRPS0.195 превращается в0.132. На семи и восьми каналах он ухудшает прогноз, как и S4 относительно MLP. Это не изолированный эффект числа каналов: меняются совместная задача и распределение параметров. Здесь фиксированная нормализация по обучающему сегменту, 2000обновлений, триseed,64истории×32прогноза; не смешиваем с предыдущими118окнами. Joint energy score оценивает весь вектор время×канал и подтверждает эти направления различий. Ridge с генерацией целого вектора остатков имеет меньшую среднюю ошибку, чем эти нейросети:0.0933/0.2870/0.0678. Эти модели не являются конкурентным forecasting benchmark. Источники: evidence.multivariate; output/extension-analysis/summary.json; output/sparse-analysis/summary.json.')
normalization=slides[10]
normalization.update(title='On ETTh1, centering removes the large head gap',backup=False,seconds=65)
normalization['notes']='65 секунд. Другой контроль — один и тот же реальный ряд температуры трансформатора, один период и118тестовых окон. GP-источник, одинаковый бюджет6000обновлений. При глобальном scaling MLP имеет CRPS0.216, S4 —0.175. Вычитаем среднее только доступной истории: обе ошибки становятся около0.098. Значит, большой разрыв зависит от обработки данных; временная архитектура не единственное объяснение. Деление ещё и на масштаб истории не даёт заметного выигрыша. Простая ridge-регрессия с исторической нормализацией получает0.0936 — лучше этих flow-моделей. Преобразование меняет весь процесс обучения, поэтому мы не выделили единственную внутреннюю причину. Это один ранее изученный период, не доказательство на всех реальных рядах. Источник: evidence.normalization и normalization_baselines.'
review=slides[11]
review.update(title='Review: completed checks and remaining gaps',backup=False,seconds=60,
 headers=['Review question','Evidence / status'],rows=[
 ['Only numerical noise?','No: checking PC3–6 keeps the same error ordering'],
 ['Does the reference know more?','Same observed history; known synthetic distribution'],
 ['Do findings repeat?','2–3 new synthetic datasets; width / normalization checks'],
 ['Fully fair and tuned?','No: history conditioning differs; broad tuning is missing']],
 bottom='Also untested: whitening regularization and S4 internal step-size settings.\nMore Euler steps checks the solver, not these modelling choices.',
 notes='60 секунд. Отвечаем на ревью явно. Первое: исключили почти шумовые направления — порядок ошибок наPC3–6 сохранился. Второе: построили эталон, который получает ту же нормализованную историю, и добавили ансамблевые прогнозные оценки; меньше ошибки поля не означает лучше прогноз. Третье: в последующих синтетических проверках использовали два-три новых набора данных и проверили ширину выхода и нормализацию. Но не всё закрыто: общего способа подачи истории, широкого подбора learning rate, sweep порога whitening и внутреннего шага S4 нет. Проверка64/128шагов Euler отвечает лишь за интегрирование прогнозов и не заменяет эти абляции. Риски whitening теперь видны и в отрицательных многомерных результатах. Числа для50/50ковариационной смеси и подробные ответы есть в отдельном документе; не претендуем на полную сходимость. Источник: docs/response_to_review.md.')
conclusion=slides[9]
review['notes']=review['notes'].replace('Числа для50/50ковариационной смеси и подробные ответы есть в отдельном документе; не претендуем на полную сходимость.', 'Эталон знает закон генерации: это диагностическая опора, а не обычная обучаемая модель.')
conclusion.update(body=[
 'A smooth starting source did not give S4 a consistent forecasting advantage.',
 'Shape diagnostics and forecast scores answer different questions; use both.',
 'Output width and normalization explain specific apparent architecture gaps.',
 'Whitening gains do not transfer to the two real multichannel tasks at this budget.'],
 bottom='Small, fixed-budget experiments; simple linear baselines remain strong. AI assistance is disclosed.',
 notes='50 секунд. Главный вывод: нет универсального победителя по архитектуре или источнику. Для оценки нужны и форма траектории, и качество прогнозного распределения. В конкретных режимах мы нашли объяснения через ширину выхода и нормализацию. Многомерные результаты ограничивают рекомендацию whitening: на реальных совместных задачах он ухудшил прогноз. Простой линейный baseline остаётся сильнее этих малых сетей. Код, сохранённые прогнозы и воспроизводимые расчёты доступны. Далее кратко называем согласованные вклады участников.')
slides=slides[:9]+[normalization,multislide,review,conclusion]+slides[12:]
# Keep the already allocated 30 seconds for a visible main-talk contribution slide.
budgets=[20,60,60,45,55,60,60,60,70,65,75,60,50]
for s,seconds in zip(slides,budgets):s['seconds']=seconds
roles=json.loads((ROOT/'docs/contributions.json').read_text())
assert roles['status']=='user_approved'
slides.pop()  # Replace the former limitations/contribution-placeholder backup.
slides.insert(13,dict(title='Team contributions',kind='table',seconds=30,backup=False,
 headers=['Team member','Contribution'],rows=[[r['name'],r['slide_summary']] for r in roles['members']],
 bottom='AI assisted implementation, analysis and writing. The authors remain responsible for the work.',
 notes='30 секунд. Егор отвечал за базовый MLP, общий контур обучения и исследование ёмкости. Кирилл — за данные, PCA, дополнительные метрики и итоговую сверку результатов. Даниил — за S4, структурированные модели и многомерный прогноз. Василий — за whitening, нормализацию, якобианы и интерпретацию диагностик. При разработке, анализе и подготовке текста использовались AI-инструменты; ответственность за работу остаётся у авторов.'))
# Clarifications from a fresh audience review; keep the frozen numerical evidence.
slides[0]['notes']='Source в названии — стартовый шум, head — блок сети, который обновляет будущую кривую. Проверяем: помогает ли гладкий старт последовательностной модели S4 по сравнению с обычной плотной MLP?'
slides[1]['notes']='Здесь две разные роли случайных кривых. GP, или гауссовский процесс, генерирует плавные синтетические данные для обучения и проверки. Отдельно мы выбираем шум, с которого модель начинает строить прогноз. Он может быть белым или тоже GP. По 32 наблюдениям получаем возможные продолжения на 16 точек. Цветные линии — четыре прогноза, чёрная — одно реализованное будущее. Совпадать они не обязаны: качество распределения проверяем на множестве историй. Гипотеза: модель S4 с временной структурой лучше выучит небольшие локальные изменения на фоне плавного общего движения. Это предположение, которое мы проверяем. Источники: docs/related_work.md; figures/provenance.json. На рисунке первая сохранённая история, без отбора по метрикам.'
slides[2]['notes']='Белый шум слева меняется независимо в каждой точке, GP-шум справа связан между соседними точками. Это только старт прогноза. При обучении мы смешиваем стартовую кривую с известным будущим и учим сеть предсказывать поправку к этой смеси, учитывая историю. При прогнозировании будущего нет: сеть многократно обновляет сразу все 16 точек, начиная с нового шума. Один случайный старт даёт один возможный прогноз. Шаги генерации не являются временем самого ряда. Источники: docs/protocol.md; Lipman et al. https://arxiv.org/abs/2210.02747. Иллюстрация источников имеет единичную дисперсию координат; это не точная ковариация основного обучения.'
slides[3].update(title='Main experiment: two heads × two noise sources',
 headers=['Velocity head','White noise','Smooth GP noise'],
 rows=[['MLP (dense)','5 training seeds','5 training seeds'],
       ['S4 (temporal)','5 training seeds','5 training seeds']],
 table_height=240,table_font_size=28,column_widths=[360,388,388],
 body=['Same data and 6,000 updates per run. Four cells, five seeds each.',
       'Additional control: GP + whitened MLP, 5 runs.'],
 bottom='Does smooth starting noise change the performance gap between MLP and S4?',
 notes='Главный эксперимент — матрица два на два. Строки: две головы поля скоростей, плотная MLP и временная S4. Столбцы: белое стартовое распределение и гладкое GP. В каждой ячейке пять обучений на одинаковых данных и с одинаковым бюджетом: всего двадцать. Смотрим, меняется ли разница между головами при смене стартового распределения. Отдельный контроль — GP с whitening перед MLP, ещё пять обучений. Whitening меняет координаты и их масштабы, но не является третьей базовой головой в матрице. Для защиты от утечек используем только обучение или доступную историю. Источник: docs/protocol.md. У MLP около 65,464 параметров, у S4 65,281. История доступна обеим моделям, но подключается по-разному; это ограничение интерпретации. PCA оценивается только на обучении, веса выбираются по validation отдельно от теста.')
slides[4]['rows'][0]=['Update-prediction MSE','Does the model predict the correct curve update?']
slides[4]['rows'][2]=['CRPS','Does the forecast distribution fit each future value?']
slides[4]['rows'][3]=['Variogram score (VS)','Does it fit differences between pairs of future times?']
slides[4]['notes']='Разделяем три вопроса. MSE проверяет поправки, которым учили сеть, на отложенных данных. Это ещё не качество будущего прогноза. R1 измеряет соседние изменения, R2 — изменения наклона. Делим показатель прогнозов на показатель данных: единица означает совпадение средней характеристики. Качество распределения оцениваем CRPS по отдельным будущим значениям и VS по разностям между моментами. У обеих метрик меньше — лучше. VS не видит общий сдвиг кривой, поэтому используем обе. Сравниваем числа внутри одного эксперимента. Источник: docs/trajectory_metrics.md.'
slides[5].update(title="All 25 runs fail our fixed error-equality test",
 bottom='Required residual/leading MSE: 0.8–1.2. All 25 fail; five-seed means shown.\nLater forecast diagnostics do not change this declared result.',
 notes='До обучения мы зафиксировали двустороннее прочтение неоднозначного критерия задания: ошибки должны быть почти равны. PCA выделяет направления, в которых меняется форма кривой. Первое содержит больше всего вариации; остаточная ошибка здесь усреднена по направлениям с третьего по шестнадцатое. Второе направление учитывается отдельно в отчёте. Правило требует, чтобы две ошибки были почти равны. Но справа ошибки намного меньше: масштабы вариации разные. Поэтому все 25 запусков не проходят правило, хотя проходят исходную проверку гладкости. Даже аналитический эталон с дополнительной информацией проваливает равенство. Это ограничение критерия, а не доказательство плохого прогноза. Дальше идёт дополнительный анализ после исходного эксперимента; результат ноль из 25 сохраняем. Источники: evidence.main.groups; docs/protocol.md. Таблица содержит средние пяти обучений; правило проверено для каждого запуска. Ошибки между источниками имеют разные шумовые уровни.')
slides[6].update(title='R1 misses changes in slope',
 bottom='GP starting noise. R1 ≈ 1 for every head, but S4 has R2 = 1.63.\nThese are post-hoc shape diagnostics, not a replacement success rule.',
 notes='Здесь дополнительная проверка формы при GP-старте. У монотонной кривой сумма соседних изменений зависит только от концов: ровный подъём и меняющий наклон подъём могут иметь одинаковый R1. R2 различает их. У S4 он около 1.63: наклон меняется сильнее, чем в данных. У MLP около единицы. Однако изгиб может быть осмысленным, поэтому высокий R2 сам по себе не доказывает плохой прогноз. Теперь проверим именно распределение будущих значений. Источник: results/trajectory-metrics/summary.json, main groups. Общий набор: 256 историй, 64 прогноза, Euler-128, пять моделей на head.')
slides[7].update(bottom='S4 intervals cross zero; whitening lowers VS in this sample.\n95% intervals over 256 histories, conditional on five fitted models per head.',
 notes='На графике разность с MLP: левее нуля лучше. Точка — средняя разность, отрезок — 95-процентный интервал. У S4 оба интервала пересекают ноль, поэтому эти данные не устанавливают лучший прогноз ни у S4, ни у MLP, несмотря на различие R2. Whitening снижает VS, но по CRPS порядок неясен. Интервалы получены повторным выбором из 256 историй при фиксированных обученных моделях. Они не учитывают новое обучение или новый генератор данных. Это исследовательский анализ после основного эксперимента. Источник: results/trajectory-metrics/summary.json, main contrasts. Fair — поправка на конечное число независимых прогнозов. Верхняя граница S4 VS положительна: +0.000020.')
slides[8]['notes']='Теперь проверяем возможные причины различий. В отдельном опыте прогнозируем 64 точки, стартуем с белого шума. Слева последняя скрытая ширина MLP равна 56, в центре — 72 при той же глубине. Расширение резко уменьшает неровность. По трём новым наборам средний R2 падает с 19.09 до 1.60; у S4 около единицы. Другой контроль с близким числом параметров даёт тот же паттерн. Это согласуется с ограничением последнего слоя, но не изолирует все эффекты архитектуры и обучения. VS тоже улучшается, CRPS меняется мало. Источники: figures/provenance.json; results/trajectory-metrics/summary.json. Показаны первая сохранённая история и четыре согласованных старта; чёрная линия — одно будущее, не цель для каждого отдельного прогноза.'
slides[9].update(bottom='GP starting noise, 118 windows from one reused chronological test segment.\nHistory-normalized ridge regression: CRPS 0.0936.',
 notes='Следующая проверка: не зависит ли преимущество архитектуры от подготовки данных? Берём реальный ряд температуры трансформатора ETTh1. Слева масштаб общий, оценённый на обучении: S4 заметно лучше MLP. В центре из каждого окна вычитаем среднее доступной истории: большой разрыв исчезает. Дополнительное деление на масштаб истории почти не помогает. Значит, вывод об архитектуре зависит от нормализации. Простая линейная ridge-регрессия здесь лучше обеих сетей. Это 118 окон одного уже изученного хронологического тестового отрезка, не широкая проверка на реальных данных. Источники: evidence.normalization, normalization_baselines; docs/followup_normalization_protocol.md. GP-старт, 6 тысяч обновлений, одинаковые единицы оценки. Нормализация меняет весь процесс обучения, не выделяя единственную внутреннюю причину.')
slides[10].update(title='Whitening hurts both tested multichannel tasks',
 notes='Последняя проверка переносимости: вместо одного значения прогнозируем вектор в каждый момент. Слева только температура, в центре семь показателей ETTh1, справа восемь валютных рядов. Модель прогнозирует каналы совместно. Whitening помогает одной температуре, но ухудшает обе многоканальные задачи. Сравниваем модели внутри каждой группы: абсолютные ошибки между задачами имеют разные масштабы. Это отдельная серия с фиксированной нормализацией по обучению и бюджетом 2 тысячи обновлений. Поэтому различие нельзя приписать только числу каналов. Оценка всего совместного вектора через energy score подтверждает эти направления различий. Ridge с добавлением сохранённых векторов остатков лучше всех трёх сетей на каждой задаче. Источники: evidence.multivariate; output/extension-analysis/summary.json; output/sparse-analysis/summary.json. Три seed, 64 истории и 32 прогноза; результаты описывают этот бюджет, без универсального вывода о многомерности.')
slides[11].update(rows=[
 ['Errors in tiny directions?','Same ordering after excluding near-zero-variance directions'],
 ['Does the oracle know more?','Yes: same observed history, plus the data-generating law'],
 ['One synthetic dataset?','Width check repeated on 3 new synthetic datasets'],
 ['Fair architecture test?','Partial: equal budgets, different use of history']],
 bottom='Still open: matched history conditioning and broader tuning,\nincluding whitening regularization and S4 internal step sizes.',
 notes='Рецензенты спросили, не объясняется ли ошибка почти отсутствующими направлениями данных. После их исключения порядок моделей сохранился. Второй вопрос — сравнение с точным эталоном: он видит ту же историю, но знает закон генерации. Поэтому это диагностический ориентир, а не обычный обучаемый конкурент. Зависимость от единственного синтетического набора частично проверили повторением опытов с шириной на новых данных. Открыты два ограничения: модели по-разному используют историю, и широкого подбора настроек нет. Мы не выдаём равный бюджет за идеально честное сравнение архитектур. Источник: docs/response_to_review.md. PC3–6 сохраняют порядок ошибок. Проверка шагов Euler относится к численному интегрированию и не заменяет настройку внутренних шагов S4.')
slides[12].update(body=[
 'The main GP-source experiment does not establish better forecasts from S4.',
 'R2 detects excess curvature that alone does not establish worse forecasts.',
 'Wider MLP output reduces roughness. Centering removes the large ETTh1 head gap.',
 'Whitening gains do not transfer to the two real multichannel tasks at this budget.'],
 notes='В основном опыте с гладким стартом мы не установили преимущество S4 по качеству прогноза. Дополнительная неровность сама по себе не означает худшее распределение. Проверки показали, насколько вывод зависит от ширины выхода и нормализации. Выигрыш whitening не перенёсся на две реальные многоканальные задачи при нашем бюджете. Это конкретные границы выводов: универсального победителя не объявляем, простые линейные модели остаются сильным ориентиром. Далее — вклады участников. Источники: results/trajectory-metrics/summary.json; evidence.normalization; evidence.multivariate.')
# Research-narrative audit: separate the original velocity test from forecast diagnostics.
slides[0].update(subtitle='Does smooth starting noise require a temporal velocity head?',notes='Source — стартовый шум, head — блок предсказания поля скоростей. Проверяем гипотезу: при гладком GP-старте временная S4 лучше выучит малые направления скорости, чем плотная MLP. Затем проверяем, что ошибки поля означают для прогнозов.')
slides[1]['title']='Forecasting a distribution of future trajectories'
slides[1]['bottom']='32 observed points → 16 future points. One random start produces one trajectory.\nRepeat with new noise to sample the forecast distribution.'
slides[2].update(title='The network predicts velocity; integration gives a forecast',figure_height=220,body_top=390,body_gap=62,body_font_size=25,
 body=['Training: mix noise ε with future y; target velocity is u = y − ε.',
       'Network input: current future x, generation time t, past h → velocity vθ.',
       'Sampling: integrate dx/dt = vθ from noise to one future trajectory.'],
 bottom='Generation time t runs from 0 to 1; it is not the time axis of the forecast.',
 notes='Белый шум имеет независимые координаты, GP-шум — коррелированные. При обучении x_t=(1−t)ε+ty: смешиваем шум и известное будущее, целевая скорость y−ε. Модель получает текущую смесь, время генерации и наблюдённую историю; выдаёт 16 скоростей, по одной на каждую будущую точку. При прогнозировании реальное будущее неизвестно: интегратор много раз вызывает поле и обновляет весь вектор. Один старт даёт одну траекторию. Источники: fm_stress/experiment.py; docs/protocol.md. Иллюстрация источников имеет единичную дисперсию; matched GP совпадает с kernel сырых данных, не с нормализованным условным будущим.')
slides[3].update(body=['Same data, 6,000 updates per run; approximately 65k parameters.',
 'Extra control: GP + MLP with PCA rotation and rescaling (5 seeds).'],
 bottom='Hypothesis: with smooth GP noise, S4 learns small velocity directions better than MLP.',
 notes='Матрица два на два: MLP и S4; белый и гладкий GP-источник. Пять обучений в каждой ячейке. MLP смешивает все координаты плотными слоями; S4 использует временную модель состояния. Гипотеза — преимущество S4 в малых направлениях скорости при гладком источнике. Дополнительные пять запусков: та же GP+MLP с вращением и масштабированием PCA-координат; перед оценкой возвращаем исходные координаты. Это контроль влияния геометрии. Данные и бюджет общие. Источник: docs/protocol.md. PCA fit только на train, нормализация только по истории, checkpoint по validation. Способ подачи истории у голов отличается; 65464 и65281параметров. Пять seed не являются пятью наборами данных.')
slides[4]=dict(title='With GP noise, two directions contain 97.8% of velocity variance',kind='chart',seconds=55,backup=False,
 chart={'categories':['Target futures','White-source velocity','GP-source velocity'],
 'series':[{'name':'Variance in the first two principal components','values':[100*E['main']['geometry'][k]['test']['top2_energy'] for k in ('target','white','gp')]}],
 'axis':'Variance captured (%)','format':'0.0','max':100,
 'caption':'Synthetic GP · history mean/scale normalization · PCA fitted on training vectors'},
 bottom='PCA finds directions of variation. The intended GP geometry passes the >90% check.\nHypothesis: dominant motion could hide errors in smaller velocity directions.',
 notes='Прежде чем сравнивать головы, проверяем геометрию. PCA находит направления совместного изменения векторов, первое объясняет больше всего вариации. Для будущих рядов первые два направления содержат98.1процента. Для целевой скорости y−ε при белом шуме85.0, при GP97.8. Именно такой контраст требовался: гладкий источник концентрирует обучение на нескольких направлениях. Поэтому проверяем ошибки и вдоль главного направления, и в остальных. PCA оценивается на обучении; столбики — тест. Источник: evidence.main.geometry. У GP validation97.8процента тоже выше90. Базисы для будущего и скоростей разные, для каждого источника свой. Проверка геометрии сама по себе не доказывает ошибку MLP.')
slides[5].update(title='GP-source residual error: MLP is lower than S4',
 headers=['Source / velocity head','MSE: PC1','Mean MSE: PC3–16','R1 ratio'],column_widths=[400,235,266,235],table_font_size=24,
 rows=[],bottom='Five-seed means. Original rule: both residual/PC1 MSE and R1 ratio in [0.8, 1.2].\nAll 25 runs pass R1; none passes the full rule.',
 notes='Это ошибки предсказания скорости, а не будущих рядов. PCA обучена на векторах целевой скорости. PC1 — главное направление, остаток — среднее поPC3–16; PC2 отдельно в отчёте. В GP-строках остаточная ошибка MLP0.0493 меньше S40.0583; whitening снижает её до0.0401. Ожидаемого преимущества S4 нет. R1 — средний модуль соседних изменений прогнозного ряда, делённый на такой же показатель данных. Все25обучений проходят эту часть. Но исходное правило одновременно требует почти равных ошибок PC1 и остатка: ни один запуск его не проходит. Ошибки направлений имеют разные масштабы; провал равенства не означает плохой прогноз. Источники: evidence.main.groups; docs/protocol.md. Сравниваем головы внутри каждого источника: между источниками различаются базисы и шумовые уровни. Исходный R1:1024траектории,Euler64; далее отдельная ансамблевая оценка.')
for source,head in [('white','mlp'),('white','s4'),('gp','mlp'),('gp','s4'),('gp','whitened_mlp')]:
 m=E['main']['groups'][source+'|'+head]['metrics']
 slides[5]['rows'].append([source.upper()+' / '+names[head],f(m['pc1_mse']['mean'],3),f(m['residual_mse']['mean'],4),f(m['roughness_ratio']['mean'],3)])
slides[6]['chart']['caption']='GP-source forecasts · 256 histories × 64 draws · original R1 retained, R2 added'
slides[6]['chart']['series'][0]['name']='R1: adjacent changes'
slides[6]['chart']['series'][1]['name']='R2: changes of slope'
slides[6]['bottom']='Ratios to observed data: 1 means matching that statistic. S4 has R2 = 1.63.\nGreater curvature alone does not establish worse forecasts.'
slides[6]['notes']='R1 оставляем, R2 добавляем постфактум. Оба считаем на едином ансамбле прогнозов. R1 измеряет модуль соседних разностей; R2 — модуль вторых разностей, то есть изменений наклона. Делим на соответствующий показатель наблюдённых рядов. У монотонных кривых с одинаковыми концами R1 одинаков, даже если меняется наклон. У S4 R1 близок к1, но R2 равен1.63. Это больше изменений наклона, а не автоматическое доказательство плохого распределения. Василий покажет прогнозные оценки тех же GP-моделей. Источник: trajectory-metrics main groups. R2 здесь не регрессионный R-squared. 256историй×64прогноза,Euler128,5обученных моделей каждой головы.'
slides[7].update(title='GP-source forecasts: shape and predictive scores differ',
 bottom='CRPS scores values at each time; VS scores differences between times. Lower is better.\nS4 intervals cross zero; whitening lowers VS. 95% intervals over fixed-model histories.')
slides[7]['notes']='Теперь оцениваем распределение прогнозов тех же GP-моделей. CRPS проверяет значения в каждый момент, VS — различия между моментами. Нужны оба: VS не замечает общий сдвиг. На графике разность с MLP, отрицательное лучше; отрезки —95-процентные интервалы. У S4 оба пересекают ноль: более высокий R2 не сопровождается установленным ухудшением прогнозов. Whitening снижает VS, по CRPS различие неубедительно. Далее переходим к отдельным проверкам: насколько сравнение голов зависит от ёмкости, нормализации и задачи. Источник: trajectory-metrics main contrasts. 256историй,64draws,Euler128;bootstrap по историям при пяти фиксированных моделях, не неопределённость нового обучения. Fair поправляет конечный размер ансамбля.'
slides[8].update(title='A wider MLP greatly reduces roughness at horizon 64',
 bottom='Separate check: 128 past / 64 future points, white noise, 12,000 updates.\nAcross 3 new datasets, widening 56 → 72 reduces mean R2 from 19.09 to 1.60.')
slides[8]['notes']='Проверяем чувствительность к ёмкости в другой задаче: история128,горизонт64,белый шум,12000обновлений. Здесь исходный MLP даёт неровные прогнозы. Увеличиваем ширину всех его скрытых слоёв с56до72 при той же глубине. Число параметров растёт: это контроль ёмкости, а не изолированная замена последнего слоя. На трёх новых наборах R2 падает с19.09до1.60,уS4около1. Другой более широкий, но менее глубокий MLP с близким числом параметров тоже улучшает форму. VS улучшается, CRPS меняется мало. Это показывает зависимость вывода от размера сети. Источники: capacity protocol; trajectory-metrics summary; figures/provenance.json. 73960→114328параметров. Первая сохранённая история и четыре согласованных старта, без отбора по результату.'
slides[9]['title']='History centering sharply reduces the ETTh1 head gap'
slides[9]['chart']['caption']='ETTh1 temperature · 48 past / 24 future points · GP noise · 6,000 updates'
slides[9]['bottom']='Scores use common units. Centering uses only the observed history.\nHistory-normalized ridge CRPS: '+f(E['normalization_baselines']['history']['forecast_means']['fair_crps'],4)+'. 118 windows from one chronological test segment.'
mc={(r['dataset'],r['source'],r['head']):r for r in E['multivariate']['cells']}
datasets=['gp_multi','etth1_uni','etth1_multi','exchange_multi']
slides[10].update(title='Whitening helps synthetic GP, but hurts real multichannel forecasts',
 chart={'categories':['Synthetic GP (3)','ETTh1 OT (1)','ETTh1 (7)','Exchange (8)'],
 'caption':'Joint forecasts (channel count) · GP noise · fixed training scaling · 2,000 updates',
 'series':[{'name':names[h],'values':[mc[d,'gp',h]['metrics']['fair_crps']['mean'] for d in datasets]} for h in names],
 'axis':'Fair CRPS (lower is better)','format':'0.000','label_size':20,'category_size':21},
 bottom='Compare heads within each task. Joint energy scores show the same directions.\nRidge + residuals CRPS: 0.193 / 0.093 / 0.287 / 0.068 (left to right).',
 notes='Переходим к совместному прогнозу нескольких величин: модель выдаёт вектор времени и каналов. Здесь четыре задачи: синтетический GP с3каналами; одна температура;7показателейETTh1;8валютных рядов. Whitening помогает синтетическому многомерномуGP и одной температуре, но ухудшает обе реальные многоканальные задачи. Поэтому дело не просто в размерности. Сравниваем головы внутри каждой группы. CRPS усредняет маргинальные оценки; energy score оценивает весь совместный вектор и подтверждает направление эффекта whitening. Простая ridge-регрессия с выборкой целого вектора остатков имеет меньший CRPS во всех четырёх задачах. Источники: evidence.multivariate; results/extension/*/baselines.json. Отдельная серия:2000обновлений,3seed,64истории×32прогноза,Euler64,фиксированный масштаб каналов поtrain. Не смешивать с предыдущей серией нормализации.')
# Baseline labels are derived from the same frozen per-task files as the report.
ridge_scores=[json.loads((ROOT/f'results/extension/{d}/baselines.json').read_text())['records']['ridge_residual_bootstrap']['means']['fair_crps'] for d in datasets]
slides[10]['bottom']='Compare heads within each task. Joint energy scores show the same directions.\nRidge + residuals CRPS: '+' / '.join(f(v,3) for v in ridge_scores)+' (left to right).'
slides[11].update(title='Checks prompted by the review',headers=['Review question','What the additional check shows'],
 rows=[['Are tiny PCA directions driving it?','GP velocity error on PC3–6: whitened MLP < MLP < S4'],
 ['Does lower field error mean better forecasts?','GP forecasts: no resolved S4–MLP score gap; whitening lowers VS'],
 ['Does the capacity effect repeat?','Wider MLP reduces roughness on 3 new synthetic datasets'],
 ['Is the architecture comparison isolated?','Equal budgets; the heads still use history differently']],
 bottom='Analytic reference: same observed history, known data-generating law.\nBroader per-head tuning and whitening regularization remain limitations.',
 notes='Ревью помогло уточнить интерпретацию. Пересчитали ошибку только поPC3–6: порядок whitenedMLP,MLP,S4сохраняется. Мы ограничили метрику, не удаляли координаты из модели. Ансамблевые оценки проверяют качество прогнозов отдельно от ошибки поля. Контроль ширины повторяется на трёх новых синтетических наборах. Аналитический эталон видит ту же нормализованную историю, но знает закон генерации; это диагностическая опора. Остаются разные способы подачи истории и неполный подбор настроек. Источник: docs/response_to_review.md; final-pca-audit. Проверка Euler не заменяет настройку внутренних шагов S4. Эффект whitening на реальных задачах не заменяет sweep floor или perturbation test.')
slides[12].update(title='Smooth starting noise alone does not justify switching to S4',body=[
 'Smooth GP noise did not produce the predicted S4 residual-error advantage.',
 'Check both trajectory shape and forecast scores: they can tell different stories.',
 'Capacity and normalization substantially change particular head comparisons.',
 'Whitening helps some tasks, but hurts both real multichannel tasks tested here.'],
 bottom='Use a simple linear baseline too. These are conclusions from fixed-budget experiments.',
 notes='Возвращаемся к вопросу: гладкий источник сам по себе не даёт оснований заменять MLP на S4. Ожидаемого преимущества S4 по остаточной ошибке скорости нет. MLP остаётся базовым вариантом; оцениваем и форму, и качество прогнозов. Прежде чем объяснять разницу архитектурой, проверяем ёмкость и нормализацию. Whitening надо проверять на конкретной задаче: он помогает даже синтетическому многомерному GP, но в реальных многоканальных задачах ухудшает результат. Простой линейный baseline обязателен для контекста. Далее вклады участников. Источники: evidence.main,capacity,normalization,multivariate.')
slides[15]['title']='Backup: velocity errors restricted to PC3–6'
slides[15]['headers']=['GP-source head','Original velocity MSE','Squared gap to analytic field']
slides[15]['bottom']='PC3–6 only. Original test: 8,192 windows; analytic comparison: 2,048 fresh windows.'
for row,label in zip(slides[16]['rows'],['MLP / random labels','MLP / mean-velocity labels','S4 / random labels','S4 / mean-velocity labels']):row[0]=label
slides[16]['bottom']='3 paired dataset/seed runs. Mean-velocity labels use the known generating law.'
slides[17]['title']='Backup: variogram correction and uncertainty'

# First-use definitions from the independent slides-only review (no notes supplied).
slides[0]['subtitle']='Source = starting noise; head = the forecasting network.\nDoes smooth noise need a network built for sequences?'
slides[1].update(bottom='Synthetic data: a Gaussian process (GP), a distribution of smooth random curves.\nValues = (value − history mean) / history standard deviation; time is in sample steps.\nOne noise draw gives one future trajectory; repeat draws to sample the distribution.',bottom_top=582,bottom_height=92,bottom_font_size=22)
slides[2]['body']=[
 'Training: noise ε, known future y → mixed curve x(t) = (1−t)ε + ty; target u = y−ε.',
 'Network: current curve x, generation time t, past h → velocity v (rates of change).',
 'Sampling: integrate dx/dt = v from noise to one future trajectory.']
slides[3].update(headers=['Velocity network (head)','White noise','Smooth GP noise'],
 rows=[['MLP: dense network','5 training seeds','5 training seeds'],['S4: state-space model','5 training seeds','5 training seeds']],
 table_height=180,table_font_size=24,body_font_size=22,body_gap=45,body_height=44,
 body=['MLP = multilayer perceptron; S4 = structured state-space sequence model.',
 '6,000 updates per run; about 65,000 parameters. GP noise correlates future positions.',
 'Whitened MLP (5 extra runs): rotate along directions of training-velocity variation;',
 'divide by their standard deviations (with a floor), then undo the transform for scoring.'],
 bottom='Hypothesis: with GP noise, S4 learns velocity directions with little variation better than MLP.',bottom_font_size=23)
slides[4]['chart'].update(caption='PCA = principal component analysis: PC1, PC2, … rank directions by decreasing variance.\nSeparate train-fitted bases for 16-point futures y and velocities u = y−ε; bars use test data.',caption_height=58,plot_top=230,plot_height=330)
slides[4]['bottom']='GP generates the data; white or GP noise starts the forecast. Values use history mean/std.\nDesign check: the first two velocity PCs should explain >90% of variance; GP passes.'
slides[4]['bottom_font_size']=23
slides[5].update(intro='MSE = mean squared error of predicted velocity v against target u (lower is better).\nErrors use training-velocity PCs; residual = mean over PC3–16, after the two leading directions.\nR1 = mean absolute change between neighbouring forecast points.\nR1 ratio = generated R1 / observed R1; 1 means matching average adjacent change.',
 intro_top=147,intro_height=115,intro_font_size=21,table_top=270,table_height=300,table_font_size=22,
 bottom='Original rule: mean MSE(PC3–16)/MSE(PC1) ∈ [0.8,1.2] AND R1 ratio ∈ [0.8,1.2].\nFive-seed means. All 25 runs pass R1; 0/25 pass both conditions.',bottom_font_size=22)
slides[6]['chart'].update(caption='R2 = mean absolute second difference: |(next − current) − (current − previous)|.\nGP-source forecasts: 256 histories, 64 generated trajectories per history.',caption_height=58,plot_top=230,plot_height=330)
slides[6]['bottom']='Ratios = generated statistic / observed statistic; 1 means a match. S4 R2 ratio = 1.63.\nGreater curvature alone does not establish worse forecasts.'
slides[6]['bottom_font_size']=23
slides[7].update(intro='CRPS = continuous ranked probability score: predicted values at each future time.\nVS = variogram score: predicted differences between times. Lower scores are better.\n“Fair” corrects for the finite number of generated trajectories per history.',
 intro_font_size=21,intro_top=148,intro_height=82,figure_top=225,figure_height=370,
 bottom='S4 intervals cross zero; whitening lowers VS. Negative differences favour that model.\n95% confidence intervals resample histories while holding the fitted models fixed.',bottom_font_size=21,bottom_top=606,bottom_height=64)
slides[8].update(bottom='Separate check: 128 past / 64 future points, white noise, 12,000 updates.\nWidth = units in each MLP hidden layer; depth and the 64-value output stay fixed.\nAcross 3 new datasets, width 56 → 72 lowers mean R2 ratio from 19.09 to 1.60.',bottom_top=581,bottom_height=94,bottom_font_size=22,figure_height=405)
slides[9]['chart'].update(categories=['Train mean/std','+ history centering','+ history mean/std'],caption='ETTh1: hourly transformer-oil temperature · 48 past / 24 future points · GP noise\n6,000 updates; scores mapped back to training-standard-deviation units.',caption_height=58,plot_top=230,plot_height=330,axis='CRPS (training-std units)',category_size=22)
slides[9].update(bottom='Centering subtracts the observed history mean; mean/std also divides by its standard deviation.\nRidge = regularized linear prediction + sampled training residuals, with history normalization.\nRidge CRPS: '+f(E['normalization_baselines']['history']['forecast_means']['fair_crps'],4)+' on the same 118 test windows.',bottom_top=581,bottom_height=94,bottom_font_size=21)
slides[10]['chart'].update(categories=['Synthetic GP (3)','Oil temperature (1)','ETTh1 (7)','Exchange rates (8)'],caption='Channels = variables: GP simulations; hourly transformer measurements; daily exchange rates.\nEach channel uses training mean/std; CRPS averages over forecast times and channels.',caption_height=58,plot_top=230,plot_height=330,axis='Mean CRPS (training-std units)',category_size=20,label_size=19)
slides[10].update(bottom='Energy score evaluates the whole future vector; lower is better and confirms the whitening trend.\nRidge + sampled training residual vectors: CRPS '+' / '.join(f(v,3) for v in ridge_scores)+' (left to right).\nParentheses give channel counts. Compare heads within a task. GP noise; 2,000 training updates.',bottom_top=581,bottom_height=94,bottom_font_size=21)
slides[11]['rows'][1][0]='Does lower velocity error mean better forecasts?'
slides[11]['bottom']='Analytic reference = mean velocity given current x, time t and past h, from the known data law.\nBroader per-head tuning and the whitening rescaling floor remain limitations.'
slides[11]['bottom_font_size']=23
slides[13].update(bottom='Jacobian: velocity sensitivity to the current curve. Kernel: GP covariance function.\nAI assisted implementation, analysis and writing; the authors remain responsible.',bottom_font_size=22)
slides[14]['body'][0]='Original rule: mean MSE(PC3–16) / MSE(PC1) must lie in [0.8, 1.2].'
slides[14]['body'][3]='Earlier analytic reference also fails: it knows the generating law and the full raw history.'
slides[15].update(intro='Both columns average squared errors over PC3–6 of training velocities.\nLeft: (v−u)² against target u=y−ε. Right: (v−v*)² against analytic mean v*=E[u|x,t,h].',intro_height=72,intro_font_size=22,table_top=240,table_height=280)
slides[16].update(intro='Random labels: target u=y−ε for one sampled future/noise pair.\nMean-velocity labels: average u over possible pairs at fixed curve x, time t and past h.',intro_height=72,intro_font_size=22,table_top=240,table_height=310,table_font_size=22)
slides[17].update(body=[
 'X[m,i] = value at time i in generated trajectory m; M = number of trajectories. For times i,j: a[m] = |X[m,i]−X[m,j]|^0.5.',
 'Time-pair weights are proportional to 1/|i−j| and sum to 1. Fair VS subtracts sample variance of a[m] divided by M for each pair.',
 '95% confidence intervals resample 256 paired histories, holding 5 fitted models per head fixed.',
 'Follow-ups: 3 paired dataset/training-seed runs; no pooled population confidence interval.'],body_font_size=25)

# Show the actual R1/R2 definitions as editable equations, before the results.
slides[5].update(intro='MSE = mean squared error of predicted velocity v against target u (lower is better).\nErrors use training-velocity PCs; residual = mean over PC3–16, after the two leading directions.',
 intro_height=54,intro_font_size=20,roughness_formula={'order':1,'top':199,'definition':'x: one forecast trajectory\nH: number of future points\ni: forecast-point index'},
 ratio_line='Reported ratio = mean R1 of generated paths / mean R1 of observed paths; 1 means a match.',ratio_top=290,
 table_top=330,table_height=240,table_font_size=21)
slides[6]['roughness_formula']={'order':2,'top':150,'definition':'x: one forecast trajectory\nH: number of future points\ni: forecast-point index'}
slides[6]['chart'].update(caption='GP-source forecasts: 256 histories, 64 generated trajectories per history.',caption_top=246,caption_height=30,plot_top=282,plot_height=280,axis='Generated / observed')
slides[6]['bottom']='Each ratio = mean statistic of generated paths / mean statistic of observed paths.\nS4 R2 ratio = 1.63. Greater curvature alone does not establish worse forecasts.'

# Shorter spoken explanations leave more room for figures and speaker handovers.
for slide,seconds in zip(slides,[20,55,55,50,55,65,55,55,60,60,70,55,45,30]):
 slide['seconds']=seconds

speech_seconds=sum(s['seconds'] for s in slides if not s['backup'])
timing={'speech_seconds':speech_seconds,'pause_transition_seconds':75,'contribution_reserve_seconds':0,
 'contribution_slide_seconds':30,'planned_total_seconds':speech_seconds+75,'hard_limit_seconds':900}
assert len(slides)==18 and timing['planned_total_seconds']<900
# Strip obsolete inline time labels; the slide allocation above is authoritative.
import re
for s in slides:s['notes']=re.sub(r'^\d+ секунд\.\s*','',s['notes'])
out=ROOT/'output/final-presentation';out.mkdir(parents=True,exist_ok=True)
(out/'content.json').write_text(json.dumps({'title':'Smooth source, unstructured head','main_minutes':speech_seconds/60,'timing':timing,'slides':slides},indent=2,ensure_ascii=False)+'\n')
(out/'speaker_notes_ru.md').write_text('# План выступления\n\n14 основных слайдов: '+str(speech_seconds)+' секунд речи (включая 30 секунд на вклады) + 75 секунд пауз и переходов = '+str(timing['planned_total_seconds'])+' секунд; лимит 900 секунд. Резервные слайды 15–18 — только на вопросы. Не читать все числа.\n\nКонтрольные точки с паузами: Егор (1–4) до 3:20, Кирилл (5–7) до 6:35, Василий (8–10) до 9:50, Даниил (11–14) до 13:25. Не читать вслух источники и технические уточнения после «Источник/Источники»; они нужны для подготовки и вопросов. При задержке сократить повтор чисел, не пропускать нормализацию, многомерность и ответы ревьюерам. Тайминг — оценка, необходима живая репетиция.\n\n'+'\n\n'.join(f"## {i+1}. {s['title'].replace(chr(10),' ')} ({s['seconds']} секунд)\n\n{s['notes']}" for i,s in enumerate(slides))+'\n')
print(len(slides),'slides;',sum(s['seconds'] for s in slides)/60,'minutes main talk')
