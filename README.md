# CNN_for_locating_corners_with_DE
Обнаружение углов на изображении – один из важнейших шагов по обработке самого изображения. Углы – основная метрика (точки интереса), которая дает представление о форме и структуре объектов на изображении. При этом, обнаружение углов является сложной задачей, которая требует высокой точности и быстродействия. Задачи, связанные со встраиванием ЦВЗ (или их извлечением) в том числе зависят от возможности точного определения углов, поэтому создание точного обнаружителя углов также улучшит различные результаты алгоритмов ЦВЗ.
Существует большое количество различных методов для обнаружения углов, таких как Harris corner detector, однако, эти методы не всегда дают хорошие результаты на изображениях с разной сложностью. Ячеистые нейронные сети (CNN) являются мощным инструментом для обработки изображений, однако они не всегда способны хорошо обнаруживать углы на изображениях. Чтобы решить эту проблему, в данной реализации используется эволюционный алгоритм Differential Evolution (DE) для оптимизации параметров CNN. 
Код прдеставлен в виде набора функции и классов.
Существует 2 файла обычный .py и версия для Jupyter Notebook