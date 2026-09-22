# FME AI Service

Servicio FastAPI que revisa el formato de documentos del proceso de desarrollo
(productos de trabajo CMMI) a partir de una imagen, usando un modelo
LayoutLMv3 afinado sobre FUNSD.

```
imagen ──► OCR (Tesseract) ──► LayoutLMv3 (entidades) ──► reglas ──► JSON
                palabras+cajas     HEADER/QUESTION/ANSWER    hallazgos
```

## Estructura

```
app/
├── main.py                 # create_app(): rutas, lifespan, manejo de errores
├── core/                   # configuracion (.env) y seguridad (X-API-Key)
├── api/                    # rutas HTTP e inyeccion de dependencias
├── schemas/                # contrato JSON expuesto a Laravel
├── domain/                 # modelos y excepciones sin dependencias externas
└── services/               # OCR, modelo, tipos de documento, reglas, revisor
training/                   # script de fine-tuning para Google Colab
tests/                      # pruebas con OCR y modelo simulados
```

## Instalacion

1. **Tesseract OCR** (Windows):
   `winget install UB-Mannheim.TesseractOCR`. El instalador silencioso solo
   trae ingles; para espanol sin permisos de administrador se usa una
   carpeta de idiomas propia (`TESSDATA_DIR`):

   ```bash
   mkdir -p models/tessdata
   cp "C:/Program Files/Tesseract-OCR/tessdata/"{eng,osd}.traineddata models/tessdata/
   curl -L -o models/tessdata/spa.traineddata https://github.com/tesseract-ocr/tessdata_fast/raw/main/spa.traineddata
   ```

2. Dependencias:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements-dev.txt
   ```

3. `copy .env.example .env` y ajustar valores.

## Modelo

El servicio siempre carga `models/fme-layoutlmv3-funsd/` (`MODEL_DIR`) y
**nunca usa internet** (`local_files_only=True`); solo requiere los archivos
locales.

| Archivo en `models/` | Uso |
| --- | --- |
| `fme-layoutlmv3-funsd/` | Modelo en uso. Hoy: modelo temporal [`nielsr/layoutlmv3-finetuned-funsd`](https://huggingface.co/nielsr/layoutlmv3-finetuned-funsd) (LayoutLMv3 afinado en FUNSD, F1 0.908, mismas etiquetas). |
| `fme-layoutlmv3-funsd.zip` | Opcional: modelo exportado desde Colab. |
| `tessdata/` | Idiomas de Tesseract (espanol e ingles). |

### Reemplazar por el modelo entrenado en Colab

1. Copiar el `fme-layoutlmv3-funsd.zip` descargado de Colab en `models/`.
2. Reiniciar el servicio.

Al iniciar, si el `.zip` es nuevo o distinto al ultimo instalado, se extrae
en una carpeta temporal y luego reemplaza `models/fme-layoutlmv3-funsd/`
(el log muestra `Instalando modelo desde ...`). Si el zip esta danado, se
conserva el modelo anterior. No hay que editar `.env` ni codigo.

Si el modelo no esta disponible el servicio arranca igual: `GET /health`
reporta `degraded` con el motivo y el endpoint responde `503`.

## Ejecucion

```bash
uvicorn app.main:app --reload
```

Documentacion interactiva en <http://127.0.0.1:8000/docs>.

**Postman**: importar
[`docs/postman/fme-ai.postman_collection.json`](docs/postman/fme-ai.postman_collection.json)
y seleccionar una imagen en el campo `imagen` (hay un ejemplo en
[`docs/samples/`](docs/samples/)).

## API

`POST /api/v1/documentos/revision-formato` (multipart, campo `imagen`;
cabecera `X-API-Key` si `API_KEY` esta configurada)

```json
{
  "cumple": true,
  "puntaje": 90,
  "tipo_detectado": "Acta de reunión",
  "resumen": "Tipo de documento: Acta de reunión. Se identificaron ...",
  "hallazgos": [
    {"severidad": "baja", "elemento": "respuestas", "mensaje": "..."}
  ]
}
```

| Codigo | Motivo |
| --- | --- |
| 401 | API key invalida o ausente |
| 413 | Imagen mayor a `MAX_UPLOAD_MB` |
| 415 | El archivo no es una imagen |
| 422 | Imagen corrupta o campo `imagen` ausente |
| 503 | Modelo u OCR no disponibles |

### Integracion con Laravel

`config/services.php`:

```php
'fme_ai' => [
    'url' => env('FME_AI_URL', 'http://127.0.0.1:8000'),
    'key' => env('FME_AI_KEY'),
    'timeout' => env('FME_AI_TIMEOUT', 60),
],
```

`app/Services/Documentos/RevisorFormato.php`:

```php
public function revisar(UploadedFile $imagen): array
{
    return Http::baseUrl(config('services.fme_ai.url'))
        ->withHeaders(array_filter([
            'X-API-Key' => config('services.fme_ai.key'),
        ]))
        ->timeout((int) config('services.fme_ai.timeout'))
        ->attach(
            'imagen',
            $imagen->get(),
            $imagen->getClientOriginalName(),
            ['Content-Type' => $imagen->getMimeType()],
        )
        ->post('/api/v1/documentos/revision-formato')
        ->throw()
        ->json();
}
```

## Pruebas y estilo

```bash
pytest
flake8
```

## Extender

- **Nuevo tipo de documento**: agregar un `DocumentType` en
  `app/services/document_types.py`.
- **Nueva regla**: escribir una funcion `RuleContext -> Finding | None` en
  `app/services/rules.py` y agregarla a `DEFAULT_RULES`.
- **Otro motor OCR**: implementar el protocolo `OcrEngine` y usarlo en
  `app/api/dependencies.py`.
