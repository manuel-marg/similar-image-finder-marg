# Similar Image Finder MARG

Sistema de control de calidad visual basado en similitud de imágenes usando **Flask + MobileNetV2 + Cosine Similarity**.

## Características

- **Álbumes dinámicos**: Crea múltiples categorías (ej. `correcto`, `defecto_rebaba`, `defecto_temperatura`) con sus imágenes de referencia
- **Webhooks por álbum**: Cada álbum tiene su URL de webhook configurable
- **Clasificación automática**: Sube una imagen de producción y el sistema la compara contra todos los álbumes
- **Disparo de webhooks**: Si la similitud supera el umbral (default 85%), hace POST automático al webhook con payload JSON estructurado
- **UI integrada**: Panel único para gestión de álbumes, subida de referencias, configuración de webhooks y prueba de clasificación

## Instalación

```bash
git clone https://github.com/manuel-marg/similar-image-finder-marg.git
cd similar-image-finder-marg
pip install -r requirements.txt
python app.py
```

Accede a `http://localhost:5000`

## API Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/` | Interfaz web |
| GET | `/api/albums` | Listar álbumes |
| POST | `/api/albums` | Crear álbum `{name, webhook_url}` |
| DELETE | `/api/albums/<name>` | Eliminar álbum |
| PUT | `/api/albums/<name>/webhook` | Actualizar webhook |
| POST | `/api/albums/<name>/upload` | Subir imágenes de referencia |
| GET | `/api/albums/<name>/images` | Listar imágenes del álbum |
| DELETE | `/api/albums/<name>/images/<file>` | Eliminar imagen |
| GET/PUT | `/api/settings` | Configuración (umbral, modelo) |
| POST | `/api/classify` | Clasificar imagen (multipart: `image`) |

## Payload de Webhook

```json
{
  "album": "defecto_rebaba",
  "display_name": "Defecto Rebaba",
  "similarity_percent": 92.34,
  "confidence_threshold": 85.0,
  "matched": true,
  "timestamp": "2026-09-13T...",
  "source": "similar-image-finder-marg",
  "event": "quality_check"
}
```

## Estructura

```
├── app.py                    # Backend Flask
├── config.json               # Álbumes + webhooks + settings
├── requirements.txt          # Dependencias
├── templates/index.html      # UI
└── static/uploads/albums/    # Imágenes de referencia
```

## Tech Stack

- Flask 3.1
- TensorFlow 2.19 (MobileNetV2)
- scikit-learn (Cosine Similarity)
- Pillow / NumPy / Requests