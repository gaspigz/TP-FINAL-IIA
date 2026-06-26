# Clasificación de Emociones Caninas con InceptionV3

Trabajo Práctico Final IIA (Opción 4: Fine-Tuning).
Autores: Santiago Bussanich, Tomás Castagnino, Gaspar Giménez.

Sistema que clasifica el estado emocional de un perro (`angry`, `happy`, `relaxed`, `sad`)
a partir de imágenes, usando Transfer Learning y Fine-Tuning sobre **InceptionV3**.

- Dataset: [Dog Emotion (Kaggle)](https://www.kaggle.com/datasets/danielshanbalico/dog-emotion) — 4 clases, 1000 imágenes cada una.
- Notebook principal: `TP_Final_IIA_Bussanich,_Castagnino,_Gimenez.ipynb`.

El notebook está preparado para correr **igual en Google Colab y en local**: la celda de
credenciales detecta el entorno automáticamente, no hay que tocar el código en ningún caso.

---

## Credenciales de Kaggle: ¿de dónde salen y por qué?

El dataset se baja con `kagglehub`, que necesita autenticarse contra Kaggle. Para eso hace
falta un **API token** propio de tu cuenta. Lo generás una sola vez:

1. Entrá a Kaggle → tu avatar → **Settings**.
2. Sección **API** → botón **Create New Token**.
3. Se descarga un archivo `kaggle.json` con esta forma:

   ```json
   { "username": "tu_usuario", "key": "una_clave_larga" }
   ```

> ⚠️ Ese archivo son tus credenciales: **no lo subas al repo**. Ya está en el `.gitignore`.

A partir de ahí, **de dónde lee esas credenciales depende del entorno**, y esa es la
decisión que resuelve el notebook solo:

| Entorno | De dónde toma las credenciales | Por qué |
| --- | --- | --- |
| **Local** | `kaggle.json` en la **raíz del proyecto** (junto a `requirements.txt`) | Todo lo que necesita el proyecto queda en la misma carpeta, sin depender de tu home. El archivo está en el `.gitignore`, así que no se sube al repo. |
| **Google Colab** | Secrets del notebook (`KAGGLE_USERNAME`, `KAGGLE_KEY`) | En Colab no hay un home persistente cómodo y subir el `.json` a la sesión es inseguro. Los *Secrets* quedan cifrados en tu cuenta y no se ven en el código. |

La celda de credenciales hace `try: from google.colab import userdata` → si estamos en Colab,
usa los Secrets; si el import falla (estás en local), lee el `kaggle.json` de la raíz del
proyecto y exporta `KAGGLE_USERNAME` / `KAGGLE_KEY` automáticamente.

---

## Opción A — Correr en local

### 1. Requisito: Python 3.12

TensorFlow todavía **no** publica wheels para Python 3.13/3.14. Si usás una versión más nueva,
`pip install tensorflow` falla con *"No matching distribution found"*. Usá **Python 3.12**.

En Mac con Homebrew:

```bash
brew install python@3.12
```

(o bajalo de [python.org](https://www.python.org/downloads/)).

### 2. Crear y activar el entorno virtual

```bash
# parado en la carpeta del proyecto
python3.12 -m venv .venv

# activar
source .venv/bin/activate        # Mac / Linux
# .venv\Scripts\activate         # Windows

python --version                 # debe decir 3.12.x
```

### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Colocar las credenciales de Kaggle

Poné el `kaggle.json` (el del paso de credenciales) en la **raíz del proyecto**, al lado de
`requirements.txt`, y restringí sus permisos:

```bash
# parado en la carpeta del proyecto
mv ~/Downloads/kaggle.json ./            # ajustá el origen si está en otra carpeta
chmod 600 kaggle.json                    # solo tu usuario puede leerlo
```

Verificá:

```bash
ls -l kaggle.json                        # debe empezar con -rw-------
```

`chmod` es un comando del sistema operativo: da igual si el `.venv` está activo o no.

> El notebook lee el `kaggle.json` desde el directorio de trabajo, por eso conviene lanzar
> Jupyter / VS Code **parado en la carpeta del proyecto** (es lo normal de todos modos).

### 5. Abrir el notebook

```bash
jupyter notebook
```

Se abre el navegador, clic en el `.ipynb` y corré las celdas con `Shift + Enter`.

> Alternativa: abrir el `.ipynb` directo en **VS Code** (extensiones Python + Jupyter) y
> elegir el `.venv` como kernel.

**Notas de local:**
- La primera vez, `kagglehub` baja el dataset a `~/.cache/kagglehub/` y después lo reutiliza.
  La ruta que imprime `path` cambia respecto a Colab, pero el código no, porque siempre se usa
  la variable `path`.
- En Mac con Apple Silicon, el paquete `tensorflow` ya trae soporte (no hace falta
  `tensorflow-macos`). Entrena en CPU: para este dataset está bien, solo no abuses de las épocas.

---

## Opción B — Correr en Google Colab

1. Subí el `.ipynb` a Colab (o abrilo desde GitHub/Drive).
2. Cargá tus credenciales como **Secrets** (icono de 🔑 en la barra izquierda). Creá dos:
   - `KAGGLE_USERNAME` → el `username` del `kaggle.json`.
   - `KAGGLE_KEY` → el `key` del `kaggle.json`.
   - Activá el toggle "Notebook access" en ambos.
3. Corré las celdas de arriba hacia abajo. La detección de entorno usa esos Secrets sola.

> En Colab no hace falta crear `.venv` ni instalar `requirements.txt`: el entorno ya viene
> con casi todo, y la primera celda hace `!pip install kagglehub`.

---

## Estructura del proyecto

```
.
├── TP_Final_IIA_Bussanich,_Castagnino,_Gimenez.ipynb   # notebook principal
├── requirements.txt                                    # dependencias (local)
├── .gitignore
└── README.md
```

El dataset, el `.venv` y los modelos entrenados (`*.keras`) **no** se versionan: se regeneran
corriendo el notebook.
