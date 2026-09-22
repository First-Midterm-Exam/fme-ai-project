# Entrenamiento de LayoutLMv3 sobre FUNSD (Google Colab)

Fine-tuning de [`microsoft/layoutlmv3-base`](https://huggingface.co/microsoft/layoutlmv3-base)
para **clasificacion de tokens** sobre [FUNSD](https://github.com/crcresearch/FUNSD):
cada palabra de un formulario escaneado se etiqueta como `HEADER`, `QUESTION`,
`ANSWER` u `O` (otro), en esquema BIO.

El servicio FastAPI usara estas entidades para evaluar el formato del documento
(etapa 2).

## Pasos en Colab

1. `Entorno de ejecucion > Cambiar tipo de entorno de ejecucion > GPU (T4)`.
2. **Celda 1** – dependencias:

   ```python
   !pip install -q "transformers>=4.46" accelerate seqeval
   ```

3. **Celda 2** – pegar el contenido completo de
   [`colab_finetune_layoutlmv3.py`](colab_finetune_layoutlmv3.py) y ejecutar.
   Para cambiar hiperparametros, editar la clase `Config` antes de ejecutar
   (por ejemplo `copy_to_drive=True` para guardar una copia en Google Drive).

Duracion aproximada en una T4: 20-30 minutos con la configuracion por defecto.

## Que hace el script

| Paso | Detalle |
| --- | --- |
| Datos | Descarga FUNSD (enlace directo del repo `crcresearch/FUNSD`, con respaldo en la fuente original). 149 docs de entrenamiento (15 se separan para validacion) y 50 de test. |
| Preprocesado | Cajas normalizadas a 0-1000; documentos largos divididos en ventanas de 512 tokens con solapamiento de 128. |
| Entrenamiento | `Trainer` de Hugging Face, evaluacion por epoca y seleccion del mejor checkpoint por F1 (seqeval) en validacion. |
| Evaluacion | Reporte por entidad sobre el split de test. |
| Exportacion | `fme-layoutlmv3-funsd.zip` con pesos, configuracion, procesador y `fme_metadata.json` (etiquetas, metricas, versiones). |

## Resultado esperado

`/content/work/fme-layoutlmv3-funsd.zip`, que se descarga automaticamente.
F1 de referencia en test: ~0.85-0.90.

El procesador se exporta con `apply_ocr=False`: en inferencia el servicio debe
aportar las palabras y cajas (OCR) por su cuenta, igual que en entrenamiento.

## Licencias

- LayoutLMv3: CC BY-NC-SA 4.0 (uso no comercial).
- FUNSD: uso no comercial / investigacion.
