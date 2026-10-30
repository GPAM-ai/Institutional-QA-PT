import collections
import numpy as np
from tqdm.auto import tqdm

def prepare_train_features(examples, tokenizer, max_length=512, stride=128):
    """
    Tokeniza las preguntas y contextos alineando la posición exacta de inicio y fin de la respuesta
    con las posiciones de los tokens generados por el tokenizer (maneja truncado por ventanas deslizantes).
    """
    # En la API de AutoTokenizer, el orden de las entradas es [Question, Context]
    tokenized_examples = tokenizer(
        examples['question'],
        examples['context'],
        max_length=max_length,
        stride=stride,
        padding='max_length',
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        truncation='only_second'
    )

    sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')
    offset_mapping = tokenized_examples['offset_mapping']

    tokenized_examples['id'] = []
    tokenized_examples['start_positions'] = []
    tokenized_examples['end_positions'] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples['input_ids'][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        tokenized_examples['id'].append(examples['id'][sample_index])
        
        answers = examples['answers'][sample_index]
        # Si no hay respuestas (pregunta imposible), el token CLS es la respuesta
        if len(answers['answer_start']) == 0:
            tokenized_examples['start_positions'].append(cls_index)
            tokenized_examples['end_positions'].append(cls_index)
            continue

        start_char = answers['answer_start'][0]
        end_char = start_char + len(answers['text'][0])

        # Encontrar los límites de tokens correspondientes al Context (que es el segundo segmento, secuencia_ids == 1)
        context_token_val = 1
        try:
            token_start_index = sequence_ids.index(context_token_val)
            token_end_index = len(sequence_ids) - sequence_ids[::-1].index(context_token_val) - 1
        except ValueError:
            token_start_index = 0
            token_end_index = len(sequence_ids) - 1

        # Detectar si la respuesta está fuera de esta ventana de contexto truncada
        if start_char < offsets[token_start_index][0] or end_char > offsets[token_end_index][1]:
            tokenized_examples['start_positions'].append(cls_index)
            tokenized_examples['end_positions'].append(cls_index)
            continue

        # Mover token_start_index al inicio de la respuesta en tokens
        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        tokenized_examples['start_positions'].append(token_start_index - 1)
        
        # Mover token_end_index al final de la respuesta en tokens
        while offsets[token_end_index][1] >= end_char:
            token_end_index -= 1
        tokenized_examples['end_positions'].append(token_end_index + 1)

    return tokenized_examples

def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size=20, max_answer_length=30):
    """
    Toma los logits de inicio y fin y mapea los mejores índices candidatos de vuelta al texto original.
    """
    all_start_logits, all_end_logits = raw_predictions

    # Aceptar ejemplos como lista de dicts o como diccionario tipo Hugging Face
    if isinstance(examples, dict):
        example_list = [
            {key: examples[key][i] for key in examples}
            for i in range(len(next(iter(examples.values()))))
        ]
        example_ids = examples['id']
    else:
        example_list = examples
        example_ids = [example['id'] for example in example_list]

    # Mapeo de ID de ejemplo a su correspondiente índice en el dataset original
    example_id_to_index = {k: i for i, k in enumerate(example_ids)}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature['id']]].append(i)

    predictions = collections.OrderedDict()

    for example_index, example in enumerate(tqdm(example_list, desc="Procesando predicciones")):
        feature_indices = features_per_example.get(example_index, [])
        valid_answers = []
        context = example['context']

        # Calcular la mejor puntuación de respuesta nula (CLS) para este ejemplo
        best_null_score = None
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # En la gran mayoría de modelos BERT, el token CLS está en el índice 0
            null_score = start_logits[0] + end_logits[0]
            if best_null_score is None or null_score > best_null_score:
                best_null_score = null_score

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]['offset_mapping']

            # Encontrar los N mejores índices de inicio y fin
            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
            
            for start_index in start_indexes:
                if start_index >= len(offset_mapping) or offset_mapping[start_index] is None or offset_mapping[start_index] == (0, 0):
                    continue
                for end_index in end_indexes:
                    if end_index >= len(offset_mapping) or offset_mapping[end_index] is None or offset_mapping[end_index] == (0, 0):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append({
                        'score': start_logits[start_index] + end_logits[end_index],
                        'text': context[start_char: end_char]
                    })

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x['score'], reverse=True)[0]
            # Si el null_score es superior al mejor score no nulo, predecir vacío
            if best_null_score is not None and best_null_score > best_answer['score']:
                best_answer = {'text': '', 'score': best_null_score}
        else:
            best_answer = {'text': '', 'score': best_null_score if best_null_score is not None else 0.0}

        predictions[example['id']] = best_answer['text']

    return predictions
