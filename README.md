# HSE Deep Learning in NLP Course
Курс по глубокому обучению в нлп для магистров компьютерной лингвистики ВШЭ

# Важные ссылки
[Страничка на сайте ВШЭ](https://www.hse.ru/edu/courses/292673762)

# Week 1 - Neural Networks

### Видео
[2019 Первая лекция вкратце](https://youtu.be/jEMdv9fW2ZA)  
[2019 Видео про производные](https://youtu.be/tZ0yCzWfbZc)  

### Слайды
[Neural Networks](https://github.com/BobaZooba/HSE-Deep-Learning-in-NLP-Course/blob/master/Week%201/Week%2001.pdf)

### Домашка
[n_layer neural network](https://github.com/BobaZooba/HSE-Deep-Learning-in-NLP-Course/blob/master/Week%201/neural_network/Homework%201.ipynb)
10 баллов, дедлайн - 23.09.2019 23:59

# Проекты
## Задачи
Если какую-то задачу вы уже делали, то нельзя выбрать ее заново.  
Задачу обязательно сделать на торче и, конечно же, используя нейросети.  
Очень важный совет: сначала сделайте бейзлайн, то есть самую простую модель, которую только можно сделать в этой задаче. Таким образом вы отладите свой пайплайн и изменений вам нужно будет делать сильно меньше. По сути только заменять саму модель и гиперпараметры. Например, задача seq2seq. Вы хотите сделать большие трансформеры с какими-нибудь хаками. Если вы будете пытаться делать сразу все, то это займет много времени, к финишу вы придете, скорее всего, поняв, что вообще ничего не работает, а еще и учится долго. Обучите сначала простую лстм с несколькими слоями и небольшим размером эмбеддингов, поймите как из нее генерить текст. То есть у вас уже будет результат, у вас уже будет читалка данных, датасет, трейн луп, генерация и вам останется только заменить саму модель.

### Image captioning
*Суть задачи:* Модель по сути seq2seq, только входная последовательность это картинка (либо побитая на области, либо просто из одного таймстемпа), на *декодере текст - описание этой картинки.  
*Максимальное количество людей:* 2  
*Датасет:* [COCO - Common Objects in Context](https://cocodataset.org/#download)  
*Сложность:* Средне-сложно  
*Ориентировочное время исполнения:* Неделя-полторы  
*Метрики:* BLEU, METEOR, ROUGE  
*Статьи в помощь:* [captioning](http://shikib.com/captioning.html)  
*Примерное решение:* Взять предобученную торчовую модель для картинок в качестве энкодера и какую-нибудь модель для текстов в качестве декодера (lstm, transformer, предобученную gpt). Также вы можете побить картинку на области и сделать аттеншин между областями и стейтами из декодера.  
*Критерий выполнения:* Сеть генерит осмысленные описания картинок из тестового набора. Одна из метрик хотя бы в половину не ниже, чем state-of-the-art  

### Transformer/GPT/BERT
*Суть задачи:* Запрогать с нуля одну из моделей. Попробовать немного поучить.  
*Максимальное количество людей:* 1  
*Датасет:* Любой с текстами, как вариант, wikitext 103  
*Сложность:* Средняя  
*Ориентировочное время исполнения:* Неделя  
*Метрики:* Perplexity  
*Статьи в помощь:* Оригинальные, [GitHub - karpathy/minGPT: A minimal PyTorch re-implementation of the OpenAI GPT (Generative Pretrained Transformer) training](https://github.com/karpathy/minGPT)  
*Примерное решение:* Просто запрогать модель, читалку данных, обучение и валидацию  
*Критерий выполнения:* Перплексия ниже хотя бы 75, адекватные предсказания  

### Finetune RU GPT - Language modeling для другого домена
*Суть задачи:* Взять русскую гпт и зафайнтюнить под определенный домен, например, детские книги, или стихи, или что-нибудь еще.  
*Максимальное количество людей:* 2  
*Датасет:* Поискать тексты на тему  
*Сложность:* Средняя  
*Ориентировочное время исполнения:* Неделя  
*Метрики:* Perplexity  
*Статьи в помощь:* те же, что и для Transformer/GPT/BERT, сами модели [GitHub - sberbank-ai/ru-gpts: Russian GPT3 models.](https://github.com/sberbank-ai/ru-gpts) (советую брать small или medium)  
*Примерное решение:* Взять готовую модель, взять тексты определенной тематики и дообучить модель на них. Также можете добавить conditioning https://arxiv.org/pdf/1909.05858.pdf или еще какие-нибудь хаки из доменной адаптации   
*Критерий выполнения:* Перплексия ниже хотя бы 50, адекватные предсказания  

###  Finetune RU GPT - Chit chat aka болталка
*Суть задачи:* Зафайнтюнить  
*Максимальное количество людей:* 2  
*Датасет:* Дам датасет из книг или можно взять ответы Мейл.ру  
*Сложность:* средне-сложно  
*Ориентировочное время исполнения:* неделя-полторы  
*Метрики:* BLEU, METEOR, ROUGE и собственная разметка на адекватность ответа  
*Статьи в помощь:* [🦄 How to build a State-of-the-Art Conversational AI with Transfer Learning | by Thomas Wolf | HuggingFace | Medium](https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313)  
*Примерное решение:* То же самое, что и предыдущая задача, только теперь подаются целые диалоги. В идеале дать понимать модели еще и то, что есть и контекст, и текущая реплика и ответ.  
*Критерий выполнения:* Перплексия ниже хотя бы 50, адекватные предсказания  

### Generative Dialog System с нуля
Тоже самое, что и предыдущая модель, но только нужно сделать все с нуля самому.  

### NLI
*Суть задачи:* Предсказывать entailment и contradiction двух текстов. Полезно для transfer learning. То есть таким образом мы можем выучить эмбеддинги предложений  
*Максимальное количество людей:* 1  
*Датасет:* https://arxiv.org/pdf/1508.05326.pdf  
*Сложность:* средняя  
*Ориентировочное время исполнения:* Неделя и меньше  
*Метрики:* R@1/K, есть в статье https://arxiv.org/pdf/1905.01969.pdf в 5-м разделе  
*Статьи в помощь:* https://arxiv.org/pdf/1508.05326.pdf, https://arxiv.org/pdf/1905.01969.pdf, https://arxiv.org/pdf/1705.02364.pdf  
*Примерное решение:* смотреть вторую статью, там есть два подхода. Еще дополнительно будет рассказана тема про сравнение двух текстов  
*Критерий выполнения:* Метрики сопоставимые с sota  

### Sentence Embeddings
*Суть задачи:* Сделать модель  
*Максимальное количество людей:* 2  
*Датасет:* Quora Question Pairs, ответы мейл.ру, [conversational-datasets/amazon_qa at master · PolyAI-LDN/conversational-datasets · GitHub](https://github.com/PolyAI-LDN/conversational-datasets/tree/master/amazon_qa)  
*Сложность:* средняя-сложная, зависит от метода  
*Ориентировочное время исполнения:* неделя  
*Метрики:* R@1/K, есть в статье https://arxiv.org/pdf/1905.01969.pdf в 5-м разделе  
*Статьи в помощь:* https://arxiv.org/pdf/1905.01969.pdf  
*Примерное решение:* смотреть первую статью, там есть два подхода. Еще дополнительно будет рассказана тема про сравнение двух текстов.  
*Критерий выполнения:* Сопоставимые метрики с аналогами   

### Retrieval Dialog System
*Суть задачи:* Сделать модель болталку (такой чатбот, с которым ты говоришь на житейские темы) с выбором ответа из пула кандидатов.    
*Максимальное количество людей:* 2  
*Датасет:* [ConvAI3: Clarifying Questions for Open-Domain Dialogue Systems (ClariQ) by DeepPavlov](http://convai.io/2018/), reddit, книги на русском (могу дать), [conversational-datasets/opensubtitles at master · PolyAI-LDN/conversational-datasets · GitHub](https://github.com/PolyAI-LDN/conversational-datasets/tree/master/opensubtitles)  
*Сложность:* средняя-сложная, зависит от метода  
*Ориентировочное время исполнения:* неделя-полторы  
*Метрики:* R@1/K, есть в статье https://arxiv.org/pdf/1905.01969.pdf в 5-м разделе  
*Статьи в помощь:* https://arxiv.org/pdf/1905.01969.pdf, тема conversational response selection  
*Примерное решение:* смотреть первую статью, там есть два подхода. Еще дополнительно будет рассказана тема про сравнение двух текстов. Хорошо бы добавить контекст в модель и переранжирование кандидатов (будет в доп материалах).  
*Критерий выполнения:* R@1/10 выше 0.5, собственная разметка адекватности ответов  

### Neural Machine Translation
*Суть задачи:* Обучить модель для генерации перевода. По сути третья домашка, но с заменой датасета и хочется увидеть хаки  
*Максимальное количество людей:* 2  
*Датасет:* Нагуглить не проблема, есть от яндекса на ru-en с одним миллионом пар  
*Сложность:* Средняя  
*Ориентировочное время исполнения:* Неделя  
*Метрики:* BLEU, ROUGE, METEOR  
*Статьи в помощь:* [Machine translation | NLP-progress](http://nlpprogress.com/english/machine_translation.html)  
*Примерное решение:* lstm или трансформер. Нагуглить хаки, например, здесь [Machine translation | NLP-progress](http://nlpprogress.com/english/machine_translation.html)  
*Критерий выполнения:* Адекватные переводы, BLEU на вашем датасете не сильно хуже аналогов  


### NER или другой sequence tagging
*Суть задачи:* Обучить сеть предсказывать является ли слово каким-либо специфичным токеном, например, локацией или персоной.
*Максимальное количество людей:* 1
*Датасет:* [Named entity recognition | NLP-progress](http://nlpprogress.com/english/named_entity_recognition.html)
*Сложность:* Низкая и средняя
*Ориентировочное время исполнения:* Неделя
*Метрики:* F1
*Статьи в помощь:* [Named entity recognition | NLP-progress](http://nlpprogress.com/english/named_entity_recognition.html)
*Примерное решение:* CNN, LSTM, BERT, etc. Почти любая модель подойдет. Хочется увидеть transfer learning.
*Критерий выполнения:* Сопоставимое с sota метрики (не сильно ниже).

### Kaggle Competition
*Суть задачи:* Взять любую kaggle задачу, согласовать со мной и решить ее.
*Максимальное количество людей:* 1
*Датасет:* [Kaggle Competitions](https://www.kaggle.com/competitions)
*Сложность:* От низкой до сложной
*Ориентировочное время исполнения:* Неделя
*Метрики:* Какие угодно
*Статьи в помощь:* Отсутствуют
*Примерное решение:* Отсутствует
*Критерий выполнения:* Входите в топ 33%

### Paper Implementation
*Суть задачи:* Взять какую-либо статью и реализовать ее.
*Максимальное количество людей:* 2, зависит от статьи
*Датасет:* Какой угодно
*Сложность:* От низкой до сложной
*Ориентировочное время исполнения:* От одного дня
*Метрики:* Какие угодно
*Статьи в помощь:* [Tracking Progress in Natural Language Processing | NLP-progress](http://nlpprogress.com)
*Примерное решение:* Unsupervised Data Augmentation или какая-нибудь статья, которую надо согласовать со мной
*Критерий выполнения:* Сопоставимые со статьей метрики

### Transfer Learning с нуля
*Суть задачи:* Сделать полный пайплайн обучения, используя transfer learning. Предтренировать модель с нуля самому. Можете воспользоваться тем, как это делалось в ULMFiT. Затем претренированную модель использовать для другой задачи.
*Максимальное количество людей:* 2
*Датасет:* Какой угодно
*Сложность:* Средняя-сложная
*Ориентировочное время исполнения:* Неделя и больше
*Метрики:* Какие угодно
*Статьи в помощь:* https://arxiv.org/pdf/1801.06146.pdf
*Примерное решение:* ULMFiT
*Критерий выполнения:* Сопоставимые метрики по этой задаче с существующими методами решения

### Data Augmentation
*Суть задачи:*Исследовать методы аугментации для задач по NLP. Выбрать задачу и решить ее, используя методы аугментации.
*Максимальное количество людей:* 1
*Датасет:* Например, для классификации
*Сложность:* От низкой до сложной
*Ориентировочное время исполнения:* Неделя
*Метрики:* Скорее всего, F1
*Статьи в помощь:* https://arxiv.org/pdf/1901.11196.pdf, https://arxiv.org/pdf/1904.12848.pdf, [Sentence-level pretraining – Борис Зубарев - YouTube](https://www.youtube.com/watch?v=Lax2Fegnh5A&t=50s)
*Примерное решение:* Взять из статей выше или нагуглить еще.
*Критерий выполнения:* Сопоставимые метрики по этой задаче с существующими методами решения, улучшение результата бейзлайна с помощью аугментаций.

### Distillation
*Суть задачи:* Обучить модель ученика решать какую-либо задачу также хорошо, как и модель учителя. Есть две модели: одна большая и с крутым перформансом, а другая быстрая и с худшими метриками. Мы хотим улучшить вторую модель за счет первой. Как мы это сделаем: мы будем обучать маленькую модель на предсказаниях от большой модели. То есть обычно в датасетах для классификации у нас таргет строго определен, то есть мы используем one hot вектор, но, на самом деле, хорошо бы иметь распределение вероятностей классов. И именно это распределение мы получаем от модели учителя, ведь модель учителя справляется с задачей лучше и может лучше понять наши данные. Это распределение мы и пытаемся повторить моделью учеником, что должно улучшить наши метрики. Также мы можем нашей моделью учителем предсказать неразмеченные данные и тем самым по сути увеличить датасет.
*Максимальное количество людей:* 1
*Датасет:* Например, для классификации. Можно взять задачу языкового моделирования или дистиллировать модель, которая дает эмбеддинги предложений. 
*Сложность:* От низкой до сложной
*Ориентировочное время исполнения:* Неделя
*Метрики:* Зависит от задачи, но по хорошему нужно проверить на целевой задаче. То есть, если вы дистиллируете модель для эмбеддингов предложений, то вам нужно проверить потом насколько хороши получились эмбеддинги у модели ученика, например, хотя бы посмотрев насколько поменялись близости между текстами.
*Статьи в помощь:* 
*Примерное решение:* Можно взять модель, которая переводит текст в вектор, для tensorflow [TensorFlow Hub](https://tfhub.dev/google/LaBSE/1) и дистиллировать ее в более простую модель на торче, используя например, среднеквадратичную ошибку. Можно взять какую-нибудь задачу классификации, дообучить берта.
*Критерий выполнения:* Модель ученик работает лучше с дистилляцией, чем без.

### Sentiment Analysis и другая классификация с предобученной моделью
*Суть задачи:* Решить задачу классификации, используя какую-либо предобученную модель, то есть сделать transfer learning.
*Максимальное количество людей:* 1
*Датасет:* Любой для классификации
*Сложность:* Низкая
*Ориентировочное время исполнения:* От одного дня
*Метрики:* Например, F1
*Статьи в помощь:* -
*Примерное решение:* Выбрать предобученную модель, навесить сверху свою и обучить на задаче.
*Критерий выполнения:* Сопоставимые с решениями-аналогами метрики

### Summarization
*Суть задачи:* Есть какая-либо статья, например, новость и есть ее заголовок. Мы хотим научить нашу сеть генерировать этот заголовок, то есть по сути задача seq2seq
*Максимальное количество людей:* 2
*Датасет:* [Генерация новостных заголовков | ВКонтакте](https://vk.com/@headline_gen-announcement)
*Сложность:* Средняя
*Ориентировочное время исполнения:* Неделя и больше, так как учится долго
*Метрики:* BLEU, ROUGE, METEOR
*Статьи в помощь:* [Генерация новостных заголовков | ВКонтакте](https://vk.com/@headline_gen-announcement)
*Примерное решение:* lstm, transformer, etc
*Критерий выполнения:* Сопоставимые метрики с референсом

### Russian SuperGLUE
*Суть задачи:* Решить минимум две задачи из этого набора.
*Максимальное количество людей:* 1
*Датасет:* [Russian SuperGLUE](https://russiansuperglue.com/tasks/), можно взять английскую версию
*Сложность:* Низкая-средняя
*Ориентировочное время исполнения:* Неделя и меньше
*Метрики:* Зависит от задачи
*Статьи в помощь:* -
*Примерное решение:* Зависит от задач
*Критерий выполнения:* Сопоставимые результаты с лидербордом

### NLP AutoML для Russian SuperGLUE
*Суть задачи:* В идеале сделать небольшой фреймворк, куда я могу передать формат данных, какая будет задача и задать метрику и задача решится. В обычном варианте просто решить все задачи из датасета
*Максимальное количество людей:* 3
*Датасет:* [Russian SuperGLUE](https://russiansuperglue.com/tasks/), можно взять английскую версию
*Сложность:* Сложная
*Ориентировочное время исполнения:* Неделя и больше, так как инженерно сложная задача.
*Метрики:* Зависят от задач
*Статьи в помощь:* -
*Примерное решение:* Если выберете, то обсудим
*Критерий выполнения:* Сопоставимые результаты с лидербордом

### Ваш проект, который надо согласовать со мной
*Суть задачи:* Your madness. Учтите, что скрапить данные дело долгое, неблагодарное и зачастую провальное, так что лучше выбрать готовый датасет. Согласуйте ваше творчество со мной
*Максимальное количество людей:* 2
*Датасет:* Любой
*Сложность:* Любая
*Ориентировочное время исполнения:* От одного дня
*Метрики:* Любые
*Статьи в помощь:* -
*Примерное решение:* Подскажу
*Критерий выполнения:* Согласуем

# Связь с преподавателем
[bobazooba@mail.ru](mailto:bobazooba@mail.ru)  
[telegram: @bobazooba](https://t.me/bobazooba)
